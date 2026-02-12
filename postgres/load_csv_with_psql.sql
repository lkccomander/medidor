-- Run this after create_db_and_table.sql
-- Example:
--   psql -h localhost -U postgres -d medidor -f data/MEDIDOR/postgres/load_csv_with_psql.sql
--
-- Update the file path below for your machine before running.
-- Note: keep \copy in one line when executed via -f to avoid parsing issues.

\copy internet_speed_samples (timestamp_utc, download_mbps, upload_mbps, ping_ms, server_name, server_country, isp, public_ip, local_hostname, machine) FROM 'C:/Projects/AI/ml.school/data/MEDIDOR/internet_speed_data.csv' WITH (FORMAT csv, HEADER true);
