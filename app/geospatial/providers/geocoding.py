"""Location search using Nominatim (OpenStreetMap's geocoding service).

Nominatim: https://nominatim.org/release-docs/latest/
Rate limits & politeness rules for the public API instance:
- Minimum interval between requests: 1 second  
- User-Agent header MUST be formatted as "AppName (<contact-email>)"
- No API key required

Overpass/Komoot Photon is used as fallback for higher rate limits.
Data copyright © OpenStreetMap contributors, under ODbL license.


"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

NOMINATIM_PUBLIC_HOST = "https://nominatim.openstreetmap.org"
PHOTON_PUBLIC_HOST  = "https://photon.komoot.io"


@dataclass  
class SearchResult:
    """Result from location search."""
    
    display_name: str        
    lat: float 
    lon: float    
    bounding_box: Optional[List[float]] = None # [south, west, north, east]
    place_rank: Optional[int] = None
    importance: Optional[float] = None  
    osm_type: Optional[str] = None
    osm_id: Optional[int] = None
    type: Optional[str] = None     


@dataclass  
class GeoBounds:  # "bounding box" 
    west: float   # min longitude     
    south: float  # min latitude   
    east: float   # max longitude     
    north: float  # max latitude  
    
    @property
    def center_lat(self) -> float:
        return (self.south + self.north) / 2
        
    @property
    def center_lon(self) -> float:  
        return (self.west + self.east) / 2
    
    @property
    def bbox_list(self) -> List[float]:
        """Return [south, west, north, east] for API params.""" 
        return [self.south, self.west, self.north, self.east] 
    
    @classmethod  
    def from_coords(cls, lat: float, lon: float, radius_km: float = 0.5) -> 'GeoBounds':
        """Create a box around coordinates with a default radius.""" 
        return cls(   
            west=lon - radius_km / 111.0,  
            south=lat - radius_km / 111.32,    
            east = lon + radius_km / 111.0,     
            north= lat + radius_km / 111.32   
        )


class NominatimClient:
    """Client for the public Nominatim geocoding API. 

    Politeness rules (from nominatim.openstreetmap.org):
    - Send User-Agent header in format "AppName (<email>)" 
    - Minimum 1 second interval between requests  
    - Max ~500-1000 queries/day on public instance (aggressive usage will be blocked)      
    """ 

    def __init__(self, user_agent: str = "Map-Creator"):
        self._user_agent = user_agent 
    
    async def search(self, query: str, limit: int = 1) -> Optional[SearchResult]:    
        """Search for a location (place name, address, or coordinates). 
         
        Returns SearchResult if found; None on failure/empty results.
        """   
        # If the input looks like lat,lon - reverse geocode
        query_stripped = query.strip()  
        
        # Check for coordinate format: "40.7128, -74.0060" 
        if "," in query_stripped and all(     
            (c.isdigit() or c == "-") for c in query_stripped.replace(" ", "")   
        ):             
            parts = [p.strip() for p in query_stripped.split(",")]  
            if len(parts) >= 2:   
                try: 
                    lat, lon = float(parts[0]), float(parts[1])        
                    
                    return await self._reverse_geocode(lat, lon)
                except ValueError:  
                    pass 
        
        # Forward geocoding — use photon first (less restrictive), then Nominatim  
        results = await self._forward_search(query, limit) 
        return results[0] if results else None   
        
    async def _reverse_geocode(self, lat: float, lon: float) -> Optional[SearchResult]:    
        """Convert lat/lon to a place name."""
        
        url = f"{NOMINATIM_PUBLIC_HOST}/reverse" 
        
        params = {"lat": lat, "lon": lon, "format": "json", "accept-language": "en"}  
            
        async with aiohttp.ClientSession() as session: 
            try: 
                resp = await asyncio.wait_for(
                    session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)),
                    timeout=12.0  
                ) 
        
                if resp.status != 200:             
                    logger.warning(f"Nominatim reverse geocoding failed with status {resp.status}") 
                    return None 
                
                data = await resp.json() 
                if not data or data.get("error"):
                    return None
                
                bbox_list = data.get("boundingbox")   
                bb = [float(b) for b in bbox_list] if bbox_list else None  
                    
                return SearchResult(  
                    display_name=data.get("display_name", ""),
                    lat=float(data["lat"]), 
                    lon=float(data["lon"]),    
                    bounding_box=bb,     
                    place_rank=data.get("place_rank"),     
                    importance=data.get("importance"),      
                    osm_type=data.get("type"),   
                    type=data.get("os_type"),     
                )  
        
            except (aiohttp.ClientError, asyncio.TimeoutError):
                logger.warning("Nominatim reverse geocoding failed") 
                return None 
            
    
    async def _forward_search(self, query: str, limit: int = 1) -> List[SearchResult]:
        """Attempt forward geocoding on each service until we get a result.""" 
        
        # Try photon first — allows higher-rate public search access.  
        results = await self._search_photon(query, limit)
        if results: 
            return results 
        
        # Fall back to nominatim (requires rate limiting)
        return await self._search_nominatim(query, limit) 
    
    async def _search_nominatim(self, query: str, limit: int = 1) -> List[SearchResult]:       
        params = {
            "q": query,  
            "format": "json", 
            "limit": str(limit),    
            "addressdetails": 1,      
            "accept-language": "en",
            "dedup": 1,        
        } 
        
        url = f"{NOMINATIM_PUBLIC_HOST}/search"

        async with aiohttp.ClientSession() as session:  
            try: 
                resp = await asyncio.wait_for(          
                    session.get(url, params=params, headers=self._headers(), timeout=aiohttp.ClientTimeout(total=10)),
                    timeout=12.0,     
                ) 
        
                if resp.status == 403 or resp.status == 429:
                    logger.warning("Nominatim rate limited")  
                    return []    
                    
                data = await resp.json() 
                
                results = [ self._parse_nominatim_result(r) for r in data ]  
                
                # Filter out nulls (failed to parse)
                return [r for r in results if r is not None]

            except (aiohttp.ClientError, asyncio.TimeoutError): 
                logger.warning("Nominatim forward search failed")  
                return [] 
    
    @staticmethod   
    def _parse_nominatim_result(result: dict) -> SearchResult:        
        bbox = result.get("boundingbox")         
        bb = [float(b) for b in bbox] if bbox else None 
            
        return SearchResult(              
            display_name=result.get("display_name", ""),
            lat=float(result["lat"]),  
            lon=float(result["lon"]),    
            bounding_box=bb, 
            place_rank=result.get("place_rank"),   
            importance=result.get("importance"),
            osm_type=result.get("osm_type"),    
            type=result.get("type"),       
        ) 
    
    @staticmethod   
    def _parse_photon_result(feat: dict) -> Optional[SearchResult]:  
        props = feat.get("properties", {})   
        bbox = feat.get("bbox") 
        bb = [float(b) for b in bbox] if bbox else None 
        
        # Coordinates are in feature geometry as a GeoJSON Point
        coords = feat["geometry"]["coordinates"]  # [lon, lat]  
        
        return SearchResult(   
            display_name=props.get("name", str(coords)),
            lat=coords[1],     
            lon=coords[0],
            bounding_box=bb,    
            place_rank=None,     
            importance=props.get("importance"),     
            osm_type=props.get("osm_key"),     
        ) 
    
    async def _search_photon(self, query: str, limit: int = 1) -> List[SearchResult]:
        """Query the photon (komoot) geocoding API.

        Photon is a fast OSM-based search engine with higher rate limits than nominatim.openstreetmap.org.

        Returns results in [SearchResult] format or [].
        """
        url = f"{PHOTON_PUBLIC_HOST}/api"
        params = {
            "q": query,
            "limit": str(min(limit, 5)),
        }

        try:
            async with aiohttp.ClientSession() as session:
                resp = await asyncio.wait_for(
                    session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)),
                    timeout=12.0
                )

                if resp.status == 200:
                    data = await resp.json()
                    results = [self._parse_photon_result(feat) for feat in data.get("features", []) or []]
                    return [r for r in results if r is not None]

        except (aiohttp.ClientError, asyncio.TimeoutError):
            logger.warning("Photon search failed")
            return []

    def _headers(self) -> Dict[str, str]:
        """Return standard http headers for API requests."""
        return {"User-Agent": self._user_agent} 
    
    async def geocode_bounds(self, query: str, center_radius_km: float = 0.5) -> Tuple[GeoBounds, Optional[float]]: 
        """Convert user-input text into geographic bounds with a confidence radius. 
        
        Returns tuple of (bounds, confidence_score) or raises ValueError on failure.   
        """ 
        result = await self.search(query, limit=1)  
        
        if not result:
            raise ValueError(f"No location found for '{query}'. Please check the spelling.")  
            
        # If result has a bounding_box from API response use that  
        bb = result.bounding_box
        
        if bb and all(b is not None for b in bb): 
            bounds = GeoBounds(
                north=bb[2] if len(bb) > 2 else result.lat,
                south=bb[1] if len(bb) > 1 else result.lat,
                east=bb[0] if len(bb) > 0 else result.lon,
                west=bb[-1] if len(bb) < 4 else result.lon          
            ) 
        else:
            bounds = GeoBounds.from_coords(result.lat, result.lon, center_radius_km)  

        return bounds, result.importance    
