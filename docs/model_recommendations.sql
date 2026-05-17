-- Run once in the Supabase SQL editor before using local_inference.py --write-supabase.
-- The inference script upserts on (ticker, context_end, forecast_day).

create table if not exists public.model_recommendations (
  id bigint generated always as identity primary key,
  run_timestamp timestamptz not null,
  ticker text not null,
  sector text,
  industry text,
  context_start date not null,
  context_end date not null,
  forecast_day smallint not null check (forecast_day between 1 and 5),
  forecast_date date not null,
  predicted_class smallint not null check (predicted_class in (0, 1, 2)),
  predicted_direction text not null check (predicted_direction in ('down', 'flat', 'up')),
  recommendation text not null check (recommendation in ('SELL', 'HOLD', 'BUY')),
  confidence double precision not null,
  prob_down double precision not null,
  prob_flat double precision not null,
  prob_up double precision not null,
  last_close double precision,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint model_recommendations_unique_prediction
    unique (ticker, context_end, forecast_day)
);

create index if not exists model_recommendations_latest_idx
  on public.model_recommendations (context_end desc, forecast_day, confidence desc);

create index if not exists model_recommendations_ticker_idx
  on public.model_recommendations (ticker, context_end desc);

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists set_model_recommendations_updated_at on public.model_recommendations;
create trigger set_model_recommendations_updated_at
before update on public.model_recommendations
for each row
execute function public.set_updated_at();

alter table public.model_recommendations enable row level security;

drop policy if exists "Public can read model recommendations" on public.model_recommendations;
create policy "Public can read model recommendations"
on public.model_recommendations
for select
to anon, authenticated
using (true);

-- Do not add a public insert/update policy. Use SUPABASE_SERVICE_ROLE_KEY or
-- SUPABASE_RECOMMENDATIONS_KEY in GitHub Actions/local automation to write rows.
