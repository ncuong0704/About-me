## 🛫 AI Travel Agent

This Streamlit app is an AI-powered travel agent that creates customized travel itineraries using OpenAI GPT-4o. It streamlines the process of researching, planning, and organizing your perfect trip—helping you discover amazing destinations and activities effortlessly.

### Features
- Search for and explore interesting travel destinations, activities, and places to stay
- Personalize your itinerary based on your preferred number of travel days
- Harness the capabilities of GPT-4o to get smart, tailored travel plans
- Download your itinerary as a calendar (.ics) file for easy import into Google Calendar, Apple Calendar, or any calendar app

### How Does It Work?

The AI Travel Agent includes two main components:
- **Researcher:** Generates search queries based on your chosen destination and travel dates, then uses SerpAPI to find relevant activities and places to stay.
- **Planner:** Uses your preferences and the research results to craft a draft itinerary, suggesting activities, dining options, and accommodations.

### How to Download and Use the Calendar Feature

After your itinerary is generated:
1. Click the "Download Itinerary as Calendar (.ics)" button next to "Generate Itinerary"
2. Save the .ics file to your computer
3. Import it into your favorite calendar application (such as Google Calendar, Apple Calendar, or Outlook)
4. Each day in your itinerary will show up as an all-day event in your calendar
5. All of the day’s details are included in the event’s description

This feature makes it simple to keep track of your plans and access your itinerary across all your devices, even when offline.

### Local vs Cloud Versions

- **travel_agent.py**: Uses OpenAI's GPT-4o to deliver high-quality itineraries (requires an OpenAI API key)
- **local_travel_agent.py**: Uses Ollama for running the language model locally, so your data stays private (requires Ollama to be installed and running)