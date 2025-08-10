import os
from typing import Dict
from fastmcp import FastMCP
from dotenv import load_dotenv
from aiohttp import ClientSession

# Load environment variables
load_dotenv()

# Initialize FastMCP
mcp = FastMCP("weather-tool")

weather_api_url = "https://wttr.in"

@mcp.tool(name = "get_weather", description="Get current weather information for a given city.")
async def get_weather(city: str) -> Dict:
    """
    Get current weather information for a given city.

    Args:
        city (str): The name of the city for which to fetch the weather information.
    """
    url = f"{weather_api_url}/{city}"
    params = {
        "format": "j1"  # JSON format
    }

    return await get_with_json_response(url, params)


async def get_with_json_response(url: str, params: Dict) -> Dict:
    async with ClientSession() as session:
        try:
            async with session.get(url, params=params) as response:
                res = await response.json()

            # You can structure the weather response as needed
            current = res.get("current_condition", [{}])[0]
            return {
                "status": "success",
                "weather": {
                    "temperature_C": current.get("temp_C"),
                    "humidity": current.get("humidity"),
                    "description": current.get("weatherDesc", [{}])[0].get("value"),
                    "wind_kmph": current.get("windspeedKmph")
                }
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }

if __name__ == "__main__":
    mcp.run()