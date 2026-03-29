from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stadium:
    team: str
    venue_name: str
    location_name: str
    latitude: float
    longitude: float
    roof_type: str
    timezone: str
    lf_distance_ft: int
    cf_distance_ft: int
    rf_distance_ft: int


STADIUMS_BY_TEAM: dict[str, Stadium] = {
    "ARI": Stadium("ARI", "Chase Field", "Phoenix, AZ", 33.4455, -112.0667, "retractable", "America/Phoenix", 330, 407, 335),
    "ATL": Stadium("ATL", "Truist Park", "Atlanta, GA", 33.8907, -84.4677, "open", "America/New_York", 335, 400, 325),
    "ATH": Stadium("ATH", "Sutter Health Park", "West Sacramento, CA", 38.5800, -121.5130, "open", "America/Los_Angeles", 330, 403, 325),
    "BAL": Stadium("BAL", "Oriole Park at Camden Yards", "Baltimore, MD", 39.2838, -76.6217, "open", "America/New_York", 333, 400, 318),
    "BOS": Stadium("BOS", "Fenway Park", "Boston, MA", 42.3467, -71.0972, "open", "America/New_York", 310, 390, 302),
    "CHC": Stadium("CHC", "Wrigley Field", "Chicago, IL", 41.9484, -87.6553, "open", "America/Chicago", 355, 400, 353),
    "CIN": Stadium("CIN", "Great American Ball Park", "Cincinnati, OH", 39.0979, -84.5082, "open", "America/New_York", 328, 404, 325),
    "CLE": Stadium("CLE", "Progressive Field", "Cleveland, OH", 41.4962, -81.6852, "open", "America/New_York", 325, 400, 325),
    "COL": Stadium("COL", "Coors Field", "Denver, CO", 39.7559, -104.9942, "open", "America/Denver", 347, 415, 350),
    "CWS": Stadium("CWS", "Rate Field", "Chicago, IL", 41.8299, -87.6338, "open", "America/Chicago", 330, 400, 335),
    "DET": Stadium("DET", "Comerica Park", "Detroit, MI", 42.3390, -83.0485, "open", "America/New_York", 345, 412, 330),
    "HOU": Stadium("HOU", "Daikin Park", "Houston, TX", 29.7572, -95.3552, "retractable", "America/Chicago", 315, 409, 326),
    "KC": Stadium("KC", "Kauffman Stadium", "Kansas City, MO", 39.0517, -94.4803, "open", "America/Chicago", 330, 410, 330),
    "LAA": Stadium("LAA", "Angel Stadium", "Anaheim, CA", 33.8003, -117.8827, "open", "America/Los_Angeles", 347, 396, 350),
    "LAD": Stadium("LAD", "Dodger Stadium", "Los Angeles, CA", 34.0739, -118.2400, "open", "America/Los_Angeles", 330, 395, 330),
    "MIA": Stadium("MIA", "loanDepot park", "Miami, FL", 25.7781, -80.2197, "retractable", "America/New_York", 344, 407, 335),
    "MIL": Stadium("MIL", "American Family Field", "Milwaukee, WI", 43.0280, -87.9712, "retractable", "America/Chicago", 344, 400, 345),
    "MIN": Stadium("MIN", "Target Field", "Minneapolis, MN", 44.9817, -93.2776, "open", "America/Chicago", 339, 411, 328),
    "NYM": Stadium("NYM", "Citi Field", "Queens, NY", 40.7571, -73.8458, "open", "America/New_York", 335, 408, 330),
    "NYY": Stadium("NYY", "Yankee Stadium", "Bronx, NY", 40.8296, -73.9262, "open", "America/New_York", 318, 408, 314),
    "PHI": Stadium("PHI", "Citizens Bank Park", "Philadelphia, PA", 39.9057, -75.1665, "open", "America/New_York", 329, 401, 330),
    "PIT": Stadium("PIT", "PNC Park", "Pittsburgh, PA", 40.4469, -80.0057, "open", "America/New_York", 325, 399, 320),
    "SD": Stadium("SD", "Petco Park", "San Diego, CA", 32.7076, -117.1569, "open", "America/Los_Angeles", 334, 396, 322),
    "SEA": Stadium("SEA", "T-Mobile Park", "Seattle, WA", 47.5914, -122.3325, "retractable", "America/Los_Angeles", 331, 401, 326),
    "SF": Stadium("SF", "Oracle Park", "San Francisco, CA", 37.7786, -122.3893, "open", "America/Los_Angeles", 339, 399, 309),
    "STL": Stadium("STL", "Busch Stadium", "St. Louis, MO", 38.6226, -90.1928, "open", "America/Chicago", 336, 400, 335),
    "TB": Stadium("TB", "George M. Steinbrenner Field", "Tampa, FL", 27.9800, -82.5076, "open", "America/New_York", 318, 408, 314),
    "TEX": Stadium("TEX", "Globe Life Field", "Arlington, TX", 32.7473, -97.0842, "retractable", "America/Chicago", 329, 407, 326),
    "TOR": Stadium("TOR", "Rogers Centre", "Toronto, ON", 43.6414, -79.3894, "retractable", "America/Toronto", 328, 400, 328),
    "WSH": Stadium("WSH", "Nationals Park", "Washington, DC", 38.8730, -77.0074, "open", "America/New_York", 337, 402, 335),
}
