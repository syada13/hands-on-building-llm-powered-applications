from typing import TypedDict
from langgraph.graph import START,END,StateGraph
from langgraph.types import Send
from langchain_openai.chat_models import AzureChatOpenAI
from dotenv import load_dotenv
import os

#Load environment variables
load_dotenv()

# Create model instance
llm = AzureChatOpenAI(
  azure_deployment="gpt-4o-mini",
  api_version="2024-08-01-preview",
  temperature=0,
  max_tokens=None,
  timeout=None,
  max_retries=2
)

#Task : User Trip Booking
#Define task using prompt

booking_prompt="""Book the trip for the customer based on the following trip booking information: {reservation}.
Please fill missing requirements with default values if not provided by the user.
Return only minimal information.
"""

#Define data schema
class TripBookingState(TypedDict):
  first_name: str
  last_name: str
  departure: str
  arrival: str
  departure_date: str
  return_date: str
  num_people: str
  hotel: str
  flight: str
  booking_details: dict
  reservations: list

#Define Map data/state schema
class BookingsState(TypedDict):
  bookings: list[str]


def generate_bookings(state:BookingsState):
  prompt = booking_prompt.format(reservation=state["reservations"])
  print(f"-------{reservation}--------")
  response=llm.invoke(prompt)
  return {"booking_details":response.content}


def continue_to_booking(state:TripBookingState) -> dict:
  return {"send":[Send("generate_bookings", {"reservation": reservation})for reservation in state["reservations"]]}

#Define a StateGraph workflow and computations
graph = StateGraph(TripBookingState)
graph.add_node("generate_bookings",generate_bookings)
graph.add_node("continue_to_booking",continue_to_booking)
graph.add_edge(START,"generate_bookings")
graph.add_edge("generate_bookings","continue_to_booking")
graph.add_edge("continue_to_booking",END)

#Compiles the state graph into a CompiledStateGraph object. The compiled graph implements the Runnable interface and can be invoked, streamed, batched, and run
app = graph.compile()

#Create graph image
image = app.get_graph().draw_mermaid_png()
image_name = "map_reduce.png"

try:
  with open(image_name,"wb") as file:
    file.write(image)
except FileNotFoundError:
  print(f"{image_name} was not found") 


#Invoke workflow/graph agent to book trip.
for reservation in ["hotel","flight","car","dinner"]:
  response = app.invoke({
    "first_name": "Suresh Kumar",
    "last_name": "Yadav",
    "departure": "New Delhi",
    "arrival": "Varanasi",
    "departure_date": "2025-07-20",
    "return_date": "2025-07-222",
    "num_people": 1,
    "hotel": "single room",
    "flight": "economy class",
    "booking_details": {},
    "reservations": [reservation] 
  })

  # Print response
  print(response)


