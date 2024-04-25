CREATE TABLE IF NOT EXISTS stocks (
        id serial primary key,
        ticker text unique
);

CREATE TABLE IF NOT EXISTS daily_bars (
            date date NOT NULL,
            sec_id integer NOT NULL,
            open double precision,
            high double precision,
            low double precision,
            close double precision,
            adj_close double precision,
            volume bigint,
            CONSTRAINT daily_bar_date_sec_uix UNIQUE (date, sec_id),
            CONSTRAINT daily_bar_sec_id_fkey FOREIGN KEY (sec_id)
                REFERENCES public.stocks (id) MATCH SIMPLE
                ON UPDATE NO ACTION
                ON DELETE NO ACTION
);



