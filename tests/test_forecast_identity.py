"""Forecast checks retain Python date equality without quantizing timestamps."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from mpas_init.netcdf import same_forecast_identity
from test_ufs_surface import generated_inputs


@pytest.mark.parametrize("zone", (None, timezone.utc, ZoneInfo("America/New_York")))
def test_identity_matches_original_datetime_comparison(zone):
    source, *_ = generated_inputs()
    instant = datetime(2024, 11, 3, 1, 30, 17, 123456, tzinfo=zone)
    source = replace(source, valid_time=instant, reference_time=instant, lead=timedelta(microseconds=1), member=3)
    for attribute in ("valid_time", "reference_time"):
        for change in ({}, {"year": 2023}, {"month": 10}, {"day": 2}, {"hour": 2}, {"minute": 31},
                       {"second": 18}, {"microsecond": 123457}, {"fold": 1}):
            other = replace(source, **{attribute: instant.replace(**change)})
            expected = (source.valid_time, source.reference_time, source.lead, source.member) == (
                other.valid_time, other.reference_time, other.lead, other.member)
            assert same_forecast_identity(source, other) == expected
    for attribute, value in (("lead", timedelta(0)), ("lead", timedelta(days=1, microseconds=1)),
                             ("member", None), ("member", 4)):
        assert not same_forecast_identity(source, replace(source, **{attribute: value}))


def test_different_timezone_equality_and_ambiguous_fold_match_python():
    source, *_ = generated_inputs()
    dates = (datetime(2024, 5, 21, 0, 1, 2, 3, tzinfo=timezone.utc),
             datetime(2024, 11, 3, 1, 30, tzinfo=ZoneInfo("America/New_York")),
             datetime(2024, 11, 3, 1, 30, fold=1, tzinfo=ZoneInfo("America/New_York")))
    for date in dates:
        for zone in (None, timezone.utc, timezone(timedelta(hours=5, minutes=30)), ZoneInfo("America/New_York")):
            other = date.replace(tzinfo=None) if zone is None else date.astimezone(zone)
            left = replace(source, valid_time=date, reference_time=date)
            right = replace(source, valid_time=other, reference_time=other)
            assert same_forecast_identity(left, right) == (date == other)
