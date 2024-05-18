import os
import requests

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain import hub
from groq import Groq

def get_location_coordinates(city_name, api_key, country_code=None, state_code=None):
    print(f"Fetching coordinates for {city_name}...")  # Debug line
    # Compose API request URL
    base_url = "http://api.openweathermap.org/geo/1.0/direct?"
    query = f"q={city_name}"
    if country_code:
        query += f",{country_code}"
    if state_code:
        query += f",{state_code}"
    query += f"&limit=1&appid={api_key}"
    complete_url = base_url + query
    # Make API call
    response = requests.get(complete_url)
    data = response.json()
    if data:
        print(f"Coordinates for {city_name}: {data[0]['lat']}, {data[0]['lon']}")
        return data[0]['lat'], data[0]['lon']
    else:
        print("No data found for specified location.")
        return None, None


def get_weather_by_coords(latitude, longitude, api_key):
    print(f"Fetching weather data for coordinates: {latitude}, {longitude}...")
    # Compose API request URL
    base_url = "https://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}lat={latitude}&lon={longitude}&appid={api_key}"
    # Make API call
    response = requests.get(complete_url)
    weather_data = response.json()
    print("Weather data received.")
    return weather_data


def generate_weather_report(city_name):
    api_key = "9248eb871bb35c059eeab82fd6bf9c3e"
    print(f"Generating weather report for {city_name}...")
    # Get city coordinates
    latitude, longitude = get_location_coordinates(city_name, api_key)
    if latitude is None or longitude is None:
        return "City not found."

    # Get weather data using coordinates
    weather_data = get_weather_by_coords(latitude, longitude, api_key)
    print(weather_data)

    # Check if weather data is valid
    if 'weather' in weather_data and 'main' in weather_data:
        # Extract necessary data
        weather_description = weather_data['weather'][0]['description']
        temperature = weather_data['main']['temp']
        temp_max = weather_data['main']['temp_max']
        wind_speed = weather_data['wind']['speed']
        wind_deg = weather_data['wind']['deg']
        client = Groq(
             api_key='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',
)
        print("Preparing data for LLM...")
        chat_completion = client.chat.completions.create(
        messages=[
        {
            "role": "user",
            "content": f"Using the following data  {weather_data } generate a concise weather description",
        }
         ],
        model="llama3-8b-8192",
     )
        response_message = chat_completion.choices[0].message.content
        return response_message
    else:
        print("Weather data not available.")
        return "Weather data not available."




