/**
 * climate-api.js — Macro Environmental Intelligence Client
 * Connects CarbonLensAI to the Environmental Impact Intelligence FastAPI microservice.
 * Incorporates Circuit-Breaker resilience with local heuristic fallback.
 * Author: Harsh Ambule (github.com/HARSH0177)
 */

const API_BASE_URL = import.meta.env.VITE_CLIMATE_API_URL || 'http://localhost:8000';
let isBackendUnavailable = false;

// Precomputed offline telemetry fallback (derived from 5.7M CPCB sensor records)
const LOCAL_CITY_FALLBACKS = {
  delhi: { name: 'Delhi', pm25_p50: 108.4, pm25_p90: 265.2, avg_ndvi: 0.28, canopy_risk: 'Moderate', dominant_source: 'Vehicular & Stubble' },
  mumbai: { name: 'Mumbai', pm25_p50: 64.2, pm25_p90: 142.0, avg_ndvi: 0.42, canopy_risk: 'Low', dominant_source: 'Industrial & Coastal' },
  bengaluru: { name: 'Bengaluru', pm25_p50: 42.1, pm25_p90: 88.5, avg_ndvi: 0.54, canopy_risk: 'Low', dominant_source: 'Urban Density' },
  nagpur: { name: 'Nagpur', pm25_p50: 58.7, pm25_p90: 124.3, avg_ndvi: 0.46, canopy_risk: 'Moderate', dominant_source: 'Thermal & Transit' },
  kolkata: { name: 'Kolkata', pm25_p50: 94.6, pm25_p90: 218.4, avg_ndvi: 0.33, canopy_risk: 'High', dominant_source: 'Solid Waste' },
  chennai: { name: 'Chennai', pm25_p50: 48.3, pm25_p90: 98.6, avg_ndvi: 0.39, canopy_risk: 'Low', dominant_source: 'Coastal Traffic' },
  hyderabad: { name: 'Hyderabad', pm25_p50: 61.5, pm25_p90: 132.8, avg_ndvi: 0.37, canopy_risk: 'Moderate', dominant_source: 'Urban Expansion' },
  pune: { name: 'Pune', pm25_p50: 55.0, pm25_p90: 115.6, avg_ndvi: 0.48, canopy_risk: 'Low', dominant_source: 'Automotive Corridor' }
};

/**
 * Fetches regional baseline environmental indicators for a given city.
 */
export async function getCityClimateBaseline(cityName = 'Delhi') {
  const key = (cityName || 'Delhi').toLowerCase().trim();

  if (isBackendUnavailable) {
    return getLocalCityFallback(key);
  }

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);

    const res = await fetch(`${API_BASE_URL}/api/v1/city-climate-risk/${encodeURIComponent(key)}`, {
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return {
      source: 'live_microservice',
      city: data.city,
      pm25_p50: data.pm25_p50_ug_m3,
      pm25_p90: data.pm25_p90_ug_m3,
      ndvi: data.satellite_ndvi_canopy_index,
      canopyRisk: data.regional_deforestation_risk,
      driver: data.dominant_emission_driver,
      provenance: data.data_provenance
    };
  } catch (err) {
    console.warn('[ClimateAPI] Backend offline or timed out; triggering circuit breaker fallback.', err.message);
    isBackendUnavailable = true;
    setTimeout(() => { isBackendUnavailable = false; }, 30000); // Retry backend after 30s
    return getLocalCityFallback(key);
  }
}

/**
 * Predicts next-day PM2.5 air-quality level using the trained XGBoost model.
 */
export async function forecastPM25(city = 'Delhi', pm25Lag1 = 85.0, month = 10) {
  if (isBackendUnavailable) {
    return localForecastFallback(pm25Lag1, month);
  }

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2500);

    const res = await fetch(`${API_BASE_URL}/api/v1/forecast/pm25`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        city,
        pm25_lag_1: parseFloat(pm25Lag1),
        month: parseInt(month, 10)
      }),
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    return localForecastFallback(pm25Lag1, month);
  }
}

function getLocalCityFallback(key) {
  const fallback = LOCAL_CITY_FALLBACKS[key] || LOCAL_CITY_FALLBACKS.delhi;
  return {
    source: 'circuit_breaker_cache',
    city: fallback.name,
    pm25_p50: fallback.pm25_p50,
    pm25_p90: fallback.pm25_p90,
    ndvi: fallback.avg_ndvi,
    canopyRisk: fallback.canopy_risk,
    driver: fallback.dominant_source,
    provenance: 'Precomputed 5.7M CPCB Baseline Cache (Offline Safe)'
  };
}

function localForecastFallback(pm25Lag1, month) {
  const winterMult = [11, 12, 1].includes(month) ? 1.22 : 0.95;
  const pred = Math.round(pm25Lag1 * 0.88 * winterMult * 10) / 10;
  return {
    status: 'fallback',
    predicted_pm25_ug_m3: pred,
    aqi_category: pred > 120 ? 'Poor' : (pred > 60 ? 'Moderate' : 'Satisfactory'),
    model_metadata: { architecture: 'Local Regression Heuristic (Offline Fallback)' }
  };
}
