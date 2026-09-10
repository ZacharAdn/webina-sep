-- The publish gate, enforced by the database.
--
-- Until now any client holding the anon key could replace the active rules, which
-- meant the app's own buttons were the only thing standing between a visitor and
-- the recommendation everyone else sees. The app now holds those buttons behind a
-- key, but an app is a client like any other: the rule belongs here.
--
-- Anon keeps reading rules and writing feedback - that is the demo. Publishing is
-- revoked and replaced by one function that checks a token first. The token is
-- derived from the project's own publishable key (see src/access.py), so the
-- deployment can compute it without anyone pasting a new secret anywhere, and a
-- reader of the repository cannot.

drop policy if exists "public insert" on public.rules;
drop policy if exists "public update" on public.rules;

-- One row, one column: the token a publish must present. Row level security with
-- no policy means anon cannot read it; the function below is security definer, so
-- it can. The value is set outside this file - it is not repository material.
create table if not exists public.publish_gate (token text primary key);
alter table public.publish_gate enable row level security;
revoke all on table public.publish_gate from anon;

create or replace function public.publish_rules(
  p_bands     jsonb,
  p_source    text,
  p_rationale text,
  p_evidence  jsonb,
  p_token     text
) returns integer
language plpgsql
security definer
set search_path = public
as $$
declare
  next_version integer;
  expected     text;
begin
  select token into expected from public.publish_gate limit 1;
  if expected is null or expected = '' or p_token is distinct from expected then
    raise exception 'publish refused: bad or missing token';
  end if;

  select coalesce(max(version), 0) + 1 into next_version from public.rules;
  update public.rules set active = false where active;
  insert into public.rules (version, bands, rationale, source, evidence, active)
  values (next_version, p_bands, p_rationale, p_source, coalesce(p_evidence, '{}'::jsonb), true);
  return next_version;
end;
$$;

revoke all on function public.publish_rules(jsonb, text, text, jsonb, text) from public;
grant execute on function public.publish_rules(jsonb, text, text, jsonb, text) to anon;
