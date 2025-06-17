import json
import os
import sys
import traceback
from datetime import datetime

import requests
from agent import *
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPEN_CAGE_API_KEY = os.getenv("OPEN_CAGE_API_KEY")
OPEN_WEATHER_API_KEY = os.getenv("OPEN_WEATHER_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

if not all(
    [GEMINI_API_KEY, OPEN_CAGE_API_KEY, OPEN_WEATHER_API_KEY, ALPHA_VANTAGE_API_KEY]
):
    print("Error: Missing required API keys. Please check your .env file.")
    print("Required API keys:")
    print("- GEMINI_API_KEY")
    print("- OPEN_CAGE_API_KEY")
    print("- OPEN_WEATHER_API_KEY")
    print("- ALPHA_VANTAGE_API_KEY")
    sys.exit(1)


try:
    # Configure the OpenAI client for Gemini
    client = AsyncOpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
except Exception as e:
    print(f"Error initializing OpenAI client: {str(e)}")
    sys.exit(1)

sys_msg = """
You are a weather-helpful assistant tasked with helping the user with weather-related queries. You must provide users with detailed responses, including all weather-related parameters like temperature, humidity, wind speed, visibility, and other available data. You have access to two tools to assist in fulfilling this task:

1. **get_coordinates**: This tool helps convert location names (like cities or regions) into geographical coordinates (latitude and longitude). You can use this tool to find the precise coordinates for a given location, which can then be passed to the get_weather tool to get the corresponding weather data. The tool uses services like OpenCageData to retrieve accurate geographical details based on input queries.

2. **get_weather**: This tool allows you to retrieve current and forecasted weather data based on specific geographical coordinates (latitude and longitude). The weather data includes parameters such as:
   - Current temperature
   - Feels-like temperature
   - Minimum and maximum temperatures
   - Atmospheric pressure
   - Humidity percentage
   - Wind speed and direction
   - Cloud coverage
   - Rainfall information (if applicable)
   - Sunrise and sunset times
   - Visibility
   - Location information (city, country)

Use these tools together to efficiently assist users with their weather-related queries.
"""


@function_tool
def get_stock_price(symbol: str) -> str:
    """Fetches the current stock price for a given symbol using Alpha Vantage."""
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY,
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "Global Quote" in data and "05. price" in data["Global Quote"]:
        price = data["Global Quote"]["05. price"]
        return f"The current price of {symbol.upper()} is ${price}."
    else:
        return f"Sorry, I couldn't find the price for {symbol.upper()}."


@function_tool
def get_coordinates(city_name: str):
    """
    This function retrieves the geographical coordinates (latitude and longitude) for a given city name
    using the OpenCage Geocoding API.

    Parameters:
    city_name (str): The name of the city for which to retrieve the coordinates.

    Returns:
    dict: A dictionary containing the latitude and longitude of the specified city. The dictionary will
    have the following structure:
        {
            "latitude": <latitude_value>,
            "longitude": <longitude_value>
        }
    In case of errors or no results, the function returns a dictionary containing an "error" key with a
    corresponding message:
        {
            "error": <error_message>
        }

    Example:
    >>> get_coordinates("London")
    {'latitude': 31.4504, 'longitude': 73.1350}

    API Key:
    This function uses the OpenCage Geocoding API, which requires an API key for usage. The key should
    be included in the `url` as a query parameter.

    Response Handling:
    - If the API request is successful (status code 200), the function extracts the latitude and longitude
      from the first result in the JSON response.
    - If no results are found or the request fails, an appropriate error message is returned.
    """

    print("Getting coordinates for:", city_name)
    url = f"https://api.opencagedata.com/geocode/v1/json?q={city_name}&key={OPEN_CAGE_API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        results = response.json().get("results", [])
        if results:
            latitude = results[0]["geometry"]["lat"]
            longitude = results[0]["geometry"]["lng"]
            return {"latitude": latitude, "longitude": longitude}
        else:
            return {"error": "No results found for the provided city."}
    else:
        return {"error": f"Request failed with status code {response.status_code}"}


@function_tool
def get_weather(latitude: str, longitude: str):
    """
    Retrieves the current weather data for a specific location using the OpenWeatherMap API
    based on the provided geographical coordinates (latitude and longitude).

    Parameters:
    latitude (str): The latitude of the location.
    longitude (str): The longitude of the location.

    Returns:
    dict: A dictionary containing current weather information if the API request is successful.
    The returned dictionary typically includes the following keys:

        - 'coord': Coordinates of the location (longitude and latitude).
            - 'lon' (float): Longitude.
            - 'lat' (float): Latitude.

        - 'weather': List containing weather condition details.
            - 'id' (int): Weather condition ID.
            - 'main' (str): Group of weather parameters (e.g., Rain, Snow).
            - 'description' (str): Detailed description of the weather (e.g., light rain).
            - 'icon' (str): Weather icon code.

        - 'base': Internal parameter used by the API.

        - 'main': Weather measurements.
            - 'temp' (float): Current temperature (in Celsius if metric units are used).
            - 'feels_like' (float): Perceived temperature.
            - 'temp_min' (float): Minimum temperature.
            - 'temp_max' (float): Maximum temperature.
            - 'pressure' (int): Atmospheric pressure at sea level (in hPa).
            - 'humidity' (int): Humidity percentage.
            - 'sea_level' (int, optional): Atmospheric pressure at sea level (if available).
            - 'grnd_level' (int, optional): Atmospheric pressure at ground level (if available).

        - 'visibility' (int): Visibility distance in meters.

        - 'wind': Wind data.
            - 'speed' (float): Wind speed in meters/second.
            - 'deg' (int): Wind direction in degrees.

        - 'rain': Rain volume (if applicable).
            - '1h' (float): Rain volume for the last 1 hour (in mm).

        - 'clouds': Cloudiness data.
            - 'all' (int): Cloudiness percentage.

        - 'dt' (int): Time of data calculation (UNIX timestamp).

        - 'sys': Additional system data.
            - 'type' (int): Internal parameter.
            - 'id' (int): Internal ID.
            - 'country' (str): Country code (ISO 3166-1 alpha-2).
            - 'sunrise' (int): Sunrise time (UNIX timestamp).
            - 'sunset' (int): Sunset time (UNIX timestamp).

        - 'timezone' (int): Shift in seconds from UTC.

        - 'id': Location ID.

        - 'name' (str): Location name.

        - 'cod' (int): Status code of the API response.

    If the request fails, a dictionary with an error message is returned:
        {
            "error": "Unable to fetch weather data. Error code: <status_code>"
        }

    Example:
    >>> get_weather("51.5072", "-0.1276")
    {
        "coord": {"lon": -0.1276, "lat": 51.5072},
        "weather": [
            {
                "id": 500,
                "main": "Rain",
                "description": "light rain",
                "icon": "10d"
            }
        ],
        "base": "stations",
        "main": {
            "temp": 17.55,
            "feels_like": 17.63,
            "temp_min": 15.99,
            "temp_max": 18.23,
            "pressure": 1003,
            "humidity": 87,
            "sea_level": 1003,
            "grnd_level": 999
        },
        "visibility": 10000,
        "wind": {"speed": 3.6, "deg": 200},
        "rain": {"1h": 0.12},
        "clouds": {"all": 75},
        "dt": 1727084244,
        "sys": {
            "type": 2,
            "id": 2011528,
            "country": "GB",
            "sunrise": 1727070458,
            "sunset": 1727114154
        },
        "timezone": 3600,
        "id": 7302135,
        "name": "Abbey Wood",
        "cod": 200
    }
    """

    print("Getting weather for:", latitude, longitude)
    # Calling the weather API with the provided latitude and longitude
    response = requests.get(
        f"https://api.openweathermap.org/data/2.5/weather?lat=31.5656822&lon=74.3141829&appid={OPEN_WEATHER_API_KEY}&units=metric"
    )

    if response.status_code == 200:
        return response.json()  # Return weather data if the call succeeds
    else:
        return {
            "error": f"Unable to fetch weather data. Error code: {response.status_code}"
        }


# Set up the agents with simpler configuration
stock_agent = Agent(
    name="stock_agent",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
    tools=[get_stock_price],
    instructions="You are a helpful assistant that provides live stock prices using the Alpha Vantage API. Always provide a response, even if you encounter an error.",
)

weather_agent = Agent(
    name="weather_agent",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
    tools=[get_coordinates, get_weather],
    instructions=sys_msg
    + "\nAlways provide a response, even if you encounter an error.",
)

triage_agent = Agent(
    name="triage_agent",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
    handoffs=[stock_agent, weather_agent],
    instructions="""You are a helpful assistant that triages the user's query and determines which agent to use.
    - If the user's query is about weather, you should handoff to the weather_agent
    - If the user's query is about stock prices, you should handoff to the stock_agent
    - Always provide a response, even if you're not sure which agent to use
    - If you're not sure, ask the user to clarify their request""",
)

# Get the directory where agent.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))
history_file = os.path.join(current_dir, "history.json")


def save_to_history(query, response):
    try:
        # Create the directory if it doesn't exist
        os.makedirs(current_dir, exist_ok=True)

        # Initialize empty history if file doesn't exist
        if not os.path.exists(history_file):
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump([], f)

        # Read existing history
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                saved_history = json.load(f)
        except json.JSONDecodeError:
            print("Warning: history.json was corrupted. Creating new history.")
            saved_history = []

        # Create new conversation entry
        conversation_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_query": query,
            "assistant_response": response,
        }

        # Append new entry
        saved_history.append(conversation_entry)

        # Write back to file with proper error handling
        try:
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(saved_history, f, indent=4, ensure_ascii=False)
            print(f"Successfully saved conversation to {history_file}")
            return True
        except PermissionError:
            print(f"Error: Permission denied when writing to {history_file}")
            return False
        except Exception as e:
            print(f"Error writing to file: {str(e)}")
            return False

    except Exception as e:
        print(f"Error in save_to_history: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback

        print(f"Full error traceback: {traceback.format_exc()}")
        return False


def clear_screen():
    """Clear the terminal screen."""
    try:
        os.system("cls" if os.name == "nt" else "clear")
    except Exception as e:
        print(f"Warning: Could not clear screen: {str(e)}")


def print_welcome():
    """Print the welcome message and instructions."""
    try:
        clear_screen()
        print("=" * 60)
        print("Welcome to the Weather and Stock Assistant! 🌤️📈")
        print("=" * 60)
        print("\nI can help you with:")
        print("1. Weather information for any location")
        print("2. Current stock prices")
        print("\nExample queries:")
        print("- What's the weather like in London?")
        print("- What's the current price of AAPL stock?")
        print("- Tell me about the weather in New York")
        print("- What's the stock price for GOOGL?")
        print(
            "\nType 'exit' to quit, 'clear' to clear the screen, or 'help' to see this message again."
        )
        print("=" * 60)
    except Exception as e:
        print(f"Error displaying welcome message: {str(e)}")


def process_user_input(user_input: str, history: list) -> tuple:
    """Process user input and return response and updated history."""
    try:
        # Add user message to history
        history.append({"role": "user", "content": user_input})

        print("\nThinking...", end="", flush=True)

        # Run the agent
        result = Runner.run_sync(
            triage_agent,
            history,
        )

        print(
            "\r" + " " * 20 + "\r", end="", flush=True
        )  # Clear the "Thinking..." message

        if not result or not hasattr(result, "final_output"):
            return (
                "I apologize, but I couldn't process your request. Please try again.",
                history,
            )

        # Update history with agent's response
        li = result.to_input_list()
        history.extend(li)

        # Save to history file
        save_to_history(user_input, result.final_output)

        return result.final_output, history

    except Exception as e:
        error_msg = f"Error processing input: {str(e)}"
        print(f"\nDebug: {error_msg}")
        print(f"Debug: {traceback.format_exc()}")
        return error_msg, history


def main():
    """Main function to run the chat interface."""
    try:
        print_welcome()
        history = []

        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == "exit":
                    print("\nGoodbye! 👋")
                    break

                if user_input.lower() == "clear":
                    clear_screen()
                    continue

                if user_input.lower() == "help":
                    print_welcome()
                    continue

                # Process the input
                response, history = process_user_input(user_input, history)
                print(f"\nAssistant: {response}")

            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                print(f"Debug: {traceback.format_exc()}")
                continue

    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        print(f"Debug: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        print(f"Debug: {traceback.format_exc()}")
        sys.exit(1)
