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


STADIUMS_BY_TEAM: dict[str, Stadium] = {
    "ARI": Stadium("ARI", "Chase Field", "Phoenix, AZ", 33.4455, -112.0667, "retractable", "America/Phoenix"),
    "ATL": Stadium("ATL", "Truist Park", "Atlanta, GA", 33.8907, -84.4677, "open", "America/New_York"),
    "ATH": Stadium("ATH", "Sutter Health Park", "West Sacramento, CA", 38.5800, -121.5130, "open", "America/Los_Angeles"),
    "BAL": Stadium("BAL", "Oriole Park at Camden Yards", "Baltimore, MD", 39.2838, -76.6217, "open", "America/New_York"),
    "BOS": Stadium("BOS", "Fenway Park", "Boston, MA", 42.3467, -71.0972, "open", "America/New_York"),
    "CHC": Stadium("CHC", "Wrigley Field", "Chicago, IL", 41.9484, -87.6553, "open", "America/Chicago"),
    "CIN": Stadium("CIN", "Great American Ball Park", "Cincinnati, OH", 39.0979, -84.5082, "open", "America/New_York"),
    "CLE": Stadium("CLE", "Progressive Field", "Cleveland, OH", 41.4962, -81.6852, "open", "America/New_York"),
    "COL": Stadium("COL", "Coors Field", "Denver, CO", 39.7559, -104.9942, "open", "America/Denver"),
    "CWS": Stadium("CWS", "Rate Field", "Chicago, IL", 41.8299, -87.6338, "open", "America/Chicago"),
    "DET": Stadium("DET", "Comerica Park", "Detroit, MI", 42.3390, -83.0485, "open", "America/New_York"),
    "HOU": Stadium("HOU", "Daikin Park", "Houston, TX", 29.7572, -95.3552, "retractable", "America/Chicago"),
    "KC": Stadium("KC", "Kauffman Stadium", "Kansas City, MO", 39.0517, -94.4803, "open", "America/Chicago"),
    "LAA": Stadium("LAA", "Angel Stadium", "Anaheim, CA", 33.8003, -117.8827, "open", "America/Los_Angeles"),
    "LAD": Stadium("LAD", "Dodger Stadium", "Los Angeles, CA", 34.0739, -118.2400, "open", "America/Los_Angeles"),
    "MIA": Stadium("MIA", "loanDepot park", "Miami, FL", 25.7781, -80.2197, "retractable", "America/New_York"),
    "MIL": Stadium("MIL", "American Family Field", "Milwaukee, WI", 43.0280, -87.9712, "retractable", "America/Chicago"),
    "MIN": Stadium("MIN", "Target Field", "Minneapolis, MN", 44.9817, -93.2776, "open", "America/Chicago"),
    "NYM": Stadium("NYM", "Citi Field", "Queens, NY", 40.7571, -73.8458, "open", "America/New_York"),
    "NYY": Stadium("NYY", "Yankee Stadium", "Bronx, NY", 40.8296, -73.9262, "open", "America/New_York"),
    "PHI": Stadium("PHI", "Citizens Bank Park", "Philadelphia, PA", 39.9057, -75.1665, "open", "America/New_York"),
    "PIT": Stadium("PIT", "PNC Park", "Pittsburgh, PA", 40.4469, -80.0057, "open", "America/New_York"),
    "SD": Stadium("SD", "Petco Park", "San Diego, CA", 32.7076, -117.1569, "open", "America/Los_Angeles"),
    "SEA": Stadium("SEA", "T-Mobile Park", "Seattle, WA", 47.5914, -122.3325, "retractable", "America/Los_Angeles"),
    "SF": Stadium("SF", "Oracle Park", "San Francisco, CA", 37.7786, -122.3893, "open", "America/Los_Angeles"),
    "STL": Stadium("STL", "Busch Stadium", "St. Louis, MO", 38.6226, -90.1928, "open", "America/Chicago"),
    "TB": Stadium("TB", "George M. Steinbrenner Field", "Tampa, FL", 27.9800, -82.5076, "open", "America/New_York"),
    "TEX": Stadium("TEX", "Globe Life Field", "Arlington, TX", 32.7473, -97.0842, "retractable", "America/Chicago"),
    "TOR": Stadium("TOR", "Rogers Centre", "Toronto, ON", 43.6414, -79.3894, "retractable", "America/Toronto"),
    "WSH": Stadium("WSH", "Nationals Park", "Washington, DC", 38.8730, -77.0074, "open", "America/New_York"),
}

