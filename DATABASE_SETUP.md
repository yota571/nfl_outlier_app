# Connect permanent prediction tracking

1. Create a Supabase project at https://supabase.com/dashboard . Keep the database password private. A free project is sufficient to start; review the provider's current limits.
2. In the Supabase SQL Editor, run supabase_setup.sql (available to download from the app's Results page).
3. Choose Connect -> Session pooler. Copy the PostgreSQL URI, substituting your database password and URL-encoding any special characters in it. Use the pooler for IPv4 hosting compatibility.
4. In Streamlit -> Manage app -> Settings -> Secrets, add:

DATABASE_URL = "your session-pooler PostgreSQL URI with ?sslmode=require"

Do not paste the real URI into chat or GitHub. Supabase also supports certificate-verified SSL; use verify-full with the provider CA where configured.
5. Restart/reload the app. Open Results and choose Test database connection. Then open Research, select a targets or rushing-attempts line, and choose Save this pregame forecast.

The first implementation tracks forecasts you explicitly save, not every unattended line. Duplicate saves at unchanged model inputs are idempotent. Records keep their original line and timestamp. Changed lines create separate records. Results are stored separately; original forecasts are not overwritten. Missing official stats remain pending, never guessed as zero. Result refresh is manual and processes finalized scored games at least 36 hours after kickoff. This does not automatically backfill historic PrizePicks lines.

The app exposes no database credentials to the browser. RLS blocks anonymous API access; the server connection performs the writes. Do not add public client policies. Track only forecasts you intend to retain. The shared hosted ledger is not per-user; anyone who can use the app can trigger its save/results controls, so keep app access private if that is undesirable.

References:
https://supabase.com/docs/guides/database/connecting-to-postgres
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
