
CREATE TABLE IF NOT EXISTS public.nfl_predictions (
 prediction_id text PRIMARY KEY,
 created_at timestamptz NOT NULL,
 kickoff timestamptz NOT NULL,
 player_id text NOT NULL,
 game_id text NOT NULL,
 market text NOT NULL,
 model_version text NOT NULL,
 payload jsonb NOT NULL
);
CREATE TABLE IF NOT EXISTS public.nfl_outcomes (
 prediction_id text PRIMARY KEY REFERENCES public.nfl_predictions(prediction_id),
 recorded_at timestamptz NOT NULL,
 actual double precision NOT NULL,
 source text NOT NULL
);
ALTER TABLE public.nfl_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.nfl_outcomes ENABLE ROW LEVEL SECURITY;
