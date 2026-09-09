-- webina-sep: the loop. Applied 2026-09-09 after the live webinar.
--
-- Three changes, one idea: the system stops learning only from someone else's
-- past and starts learning from its own present.
--   1. predictions gains an outcome -- what actually happened, written later.
--   2. feedback  -- a person's verdict on one recommendation, and a better one.
--   3. rules     -- the bands the recommender reads, versioned, so the learner
--                   can write a new version and the recommender picks it up.
-- Every statement is idempotent.

alter table public.predictions
  add column if not exists outcome    text,
  add column if not exists outcome_at timestamptz;

drop policy if exists "public update" on public.predictions;
create policy "public update" on public.predictions
  for update to anon using (true) with check (true);

create table if not exists public.feedback (
  id                 bigint generated always as identity primary key,
  prediction_id      bigint references public.predictions(id),
  record_id          text not null,
  probability        numeric,
  recommended_action text,
  verdict            text not null check (verdict in ('right', 'wrong')),
  actual_outcome     text,            -- 'stayed' | 'left' | 'unknown'
  better_action      text,
  note               text,
  rules_version      integer,
  created_at         timestamptz default now()
);

alter table public.feedback enable row level security;
drop policy if exists "public read"   on public.feedback;
drop policy if exists "public insert" on public.feedback;
create policy "public read"   on public.feedback for select to anon using (true);
create policy "public insert" on public.feedback for insert to anon with check (true);

create table if not exists public.rules (
  id         bigint generated always as identity primary key,
  version    integer not null unique,
  bands      jsonb   not null,        -- [{"name","min","action"}], highest min first
  rationale  text,
  source     text not null,           -- 'seed' | 'learner:rules' | 'learner:groq'
  evidence   jsonb,                   -- the feedback digest the learner saw
  active     boolean not null default false,
  created_at timestamptz default now()
);

alter table public.rules enable row level security;
drop policy if exists "public read"   on public.rules;
drop policy if exists "public insert" on public.rules;
drop policy if exists "public update" on public.rules;
create policy "public read"   on public.rules for select to anon using (true);
create policy "public insert" on public.rules for insert to anon with check (true);
create policy "public update" on public.rules for update to anon using (true) with check (true);

-- Version 1 is the ladder.toml bands as they were on stage on 9.9.
insert into public.rules (version, bands, rationale, source, active)
select 1,
       '[{"name":"high","min":0.60,"action":"Contact this week and make a concrete offer"},
         {"name":"medium","min":0.35,"action":"Add to next month''s outreach list"},
         {"name":"low","min":0.00,"action":"No action"}]'::jsonb,
       'The three bands from ladder.toml, as run live on 2026-09-09.',
       'seed', true
where not exists (select 1 from public.rules where version = 1);
