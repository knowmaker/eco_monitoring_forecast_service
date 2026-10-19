BEGIN;

CREATE TABLE public.gas_hourly_features (
    monitoring_post_id BIGINT NOT NULL REFERENCES public.monitoring_posts(id),
    substance_code TEXT NOT NULL,
    bucket_start TIMESTAMPTZ NOT NULL,
    raw_hourly_mean DOUBLE PRECISION,
    raw_hourly_median DOUBLE PRECISION,
    filtered_hourly_mean DOUBLE PRECISION,
    hourly_min DOUBLE PRECISION,
    hourly_max DOUBLE PRECISION,
    hourly_p95 DOUBLE PRECISION,
    hourly_std DOUBLE PRECISION,
    samples_count INTEGER NOT NULL CHECK (samples_count >= 0),
    refreshed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (monitoring_post_id, substance_code, bucket_start)
);

CREATE INDEX idx_forecast_hourly_features_substance_bucket
    ON public.gas_hourly_features (substance_code, bucket_start DESC, monitoring_post_id);

CREATE TABLE public.external_weather_hourly (
    monitoring_post_id BIGINT NOT NULL REFERENCES public.monitoring_posts(id),
    bucket_start TIMESTAMPTZ NOT NULL,
    data_kind TEXT NOT NULL CHECK (data_kind IN ('historical_forecast', 'live_forecast')),
    provider TEXT NOT NULL DEFAULT 'open-meteo',
    model_latitude DOUBLE PRECISION,
    model_longitude DOUBLE PRECISION,
    air_temp DOUBLE PRECISION,
    air_hum DOUBLE PRECISION,
    atm_press DOUBLE PRECISION,
    hor_win_spd DOUBLE PRECISION,
    hor_win_dir DOUBLE PRECISION,
    precipitation DOUBLE PRECISION,
    cloud_cover DOUBLE PRECISION,
    cloud_cover_low DOUBLE PRECISION,
    wind_speed_100m DOUBLE PRECISION,
    wind_direction_100m DOUBLE PRECISION,
    wind_gusts_10m DOUBLE PRECISION,
    boundary_layer_height DOUBLE PRECISION,
    shortwave_radiation DOUBLE PRECISION,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (monitoring_post_id, bucket_start, data_kind)
);

CREATE INDEX idx_external_weather_kind_bucket
    ON public.external_weather_hourly (data_kind, bucket_start DESC, monitoring_post_id);

-- Exactly one current artifact for all gases and stations.
CREATE TABLE public.current_model (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    artifact BYTEA NOT NULL,
    feature_columns JSONB NOT NULL,
    categorical_features JSONB NOT NULL DEFAULT '[]'::jsonb,
    train_start TIMESTAMPTZ NOT NULL,
    train_end TIMESTAMPTZ NOT NULL,
    rows_count INTEGER NOT NULL CHECK (rows_count > 0),
    metrics JSONB NOT NULL,
    residual_intervals JSONB NOT NULL,
    trained_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE public.gas_predictions (
    monitoring_post_id BIGINT NOT NULL REFERENCES public.monitoring_posts(id),
    substance_code TEXT NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data_cutoff TIMESTAMPTZ NOT NULL,
    target_start TIMESTAMPTZ NOT NULL,
    target_end TIMESTAMPTZ NOT NULL,
    predicted_value DOUBLE PRECISION,
    lower_bound DOUBLE PRECISION,
    upper_bound DOUBLE PRECISION,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'ready', 'failed', 'unavailable')),
    error_message TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (monitoring_post_id, substance_code, target_start)
);

CREATE INDEX idx_forecast_predictions_concentration_map
    ON public.gas_predictions (substance_code, target_start DESC, monitoring_post_id);

CREATE TABLE public.gas_concentration_frames (
    id BIGSERIAL PRIMARY KEY,
    substance_code TEXT NOT NULL,
    hour_start TIMESTAMPTZ NOT NULL,
    data_kind TEXT NOT NULL CHECK (data_kind IN ('analysis', 'forecast')),
    payload BYTEA NOT NULL,
    content_sha256 TEXT NOT NULL CHECK (length(content_sha256) = 64),
    cell_count INTEGER NOT NULL CHECK (cell_count > 0),
    cell_size_meters INTEGER NOT NULL CHECK (cell_size_meters > 0),
    south DOUBLE PRECISION NOT NULL,
    west DOUBLE PRECISION NOT NULL,
    north DOUBLE PRECISION NOT NULL,
    east DOUBLE PRECISION NOT NULL,
    value_min DOUBLE PRECISION NOT NULL,
    value_max DOUBLE PRECISION NOT NULL,
    source_station_count INTEGER NOT NULL CHECK (source_station_count >= 1),
    wind_speed DOUBLE PRECISION,
    wind_direction DOUBLE PRECISION,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (substance_code, hour_start, data_kind),
    CHECK (south < north),
    CHECK (west < east),
    CHECK (value_min <= value_max)
);

CREATE INDEX idx_gas_concentration_frames_timeline
    ON public.gas_concentration_frames (substance_code, hour_start DESC, data_kind);

COMMIT;
