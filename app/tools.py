"""Toolbelt assembly for agents with Alibaba Cloud integration.

Collects third-party tools and local tools (RAG with OSS + DashScope, Tavily, weather forecast)
into a single list that graphs can bind to Qwen LLM.
"""
from __future__ import annotations

from typing import List

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from langchain_core.tools import tool

# Setup the Open-Meteo API client with cache and retry
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

PROVINCES = {
    "Laguna": {"lat": 14.1702, "lon": 121.2418},
    "Nueva Ecija": {"lat": 15.5784, "lon": 121.1113},
    "Isabela": {"lat": 16.9754, "lon": 121.8107}
}

@tool
def get_weather_forecast(location: str, days_ahead: int = 7) -> str:
    """Get weather forecast for Philippine rice-producing provinces."""

    location_upper = location.upper()
    matching_province = None
    
    for province, coords in PROVINCES.items():
        if province.upper() in location_upper or location_upper in province.upper():
            matching_province = province
            break
    
    if not matching_province:
        return f"Location '{location}' not found. Available provinces: {', '.join(PROVINCES.keys())}"
    
    coords = PROVINCES[matching_province]
    
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
            "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation"],
            "timezone": "Asia/Singapore",
            "past_days": 7,
            "forecast_days": min(days_ahead, 16)
        }
        
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        # Process hourly data
        hourly = response.Hourly()
        hourly_data = pd.DataFrame({
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
            "precipitation": hourly.Variables(2).ValuesAsNumpy()
        })

        # Process daily data
        daily = response.Daily()
        daily_data = pd.DataFrame({
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left"
            ),
            "temperature_2m_max": daily.Variables(0).ValuesAsNumpy(),
            "temperature_2m_min": daily.Variables(1).ValuesAsNumpy(),
            "precipitation_sum": daily.Variables(2).ValuesAsNumpy()
        })

        # Summarize
        past_rain = daily_data["precipitation_sum"][:7].sum()
        forecast_rain = daily_data["precipitation_sum"][7:].sum()
        forecast_tmax = daily_data["temperature_2m_max"][7:].max()
        forecast_tmin = daily_data["temperature_2m_min"][7:].min()
        forecast_tavg = daily_data[["temperature_2m_max", "temperature_2m_min"]][7:].mean().mean()
        forecast_humidity = hourly_data["relative_humidity_2m"][-days_ahead*24:].mean()

        return f"""Weather Summary for {matching_province}, Philippines

**Past 7 Days:**
- Total rainfall: {past_rain:.1f} mm

**Next {min(days_ahead, 16)} Days Forecast:**
- Expected rainfall: {forecast_rain:.1f} mm
- Temperature range: {forecast_tmin:.1f}°C (min) to {forecast_tmax:.1f}°C (max)
- Average temperature: {forecast_tavg:.1f}°C
- Average relative humidity: {forecast_humidity:.1f}%

**For Disease Risk Assessment:**
Use the retrieve_crop_information tool to query Alibaba Cloud OSS about rice diseases associated with these conditions:
- Temperature: {forecast_tavg:.1f}°C
- Humidity: {forecast_humidity:.1f}%
- Rainfall: {forecast_rain:.1f} mm over {min(days_ahead, 16)} days

Source: [Open-Meteo Weather API]"""
        
    except Exception as e:
        return f"Error fetching weather data for {matching_province}: {str(e)}"


def get_tool_belt_alibaba() -> List:
    """
    Return the list of tools with Alibaba Cloud integration.
    Uses OSS + DashScope embeddings for RAG with Qwen LLM.
    """
    from .rag_alibaba import (
        retrieve_crop_information,
        retrieve_insurance_information
    )
    
    tavily_tool = TavilySearchResults(max_results=5)
    
    return [
        retrieve_crop_information,      # OSS + DashScope RAG
        retrieve_insurance_information, # OSS + DashScope RAG
        tavily_tool,                    # Web search
        PubmedQueryRun(),               # PubMed search
        ArxivQueryRun(),                # ArXiv search
        get_weather_forecast            # Weather forecast
    ]


# Alias for backward compatibility
get_tool_belt = get_tool_belt_alibaba