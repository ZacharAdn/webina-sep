-- Where the publish token lives.
--
-- One row, one column. Row level security with no policy at all means the anon
-- role cannot read it; publish_rules is security definer, so it can. The value
-- itself is set outside this file - it is not repository material.

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
