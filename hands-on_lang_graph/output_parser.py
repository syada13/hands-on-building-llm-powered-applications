from langchain_google_vertex import ChatVertexAI
# I am skipping environment variables load on purpose
llm = ChatVertexAI(model="gemini-1.5")
job_description: str = ""
prompt_template = (
  "Given a job description, decide whether it suites a junior Java developer."
  "\nJOB DESCRIPTION:\n{job_description}\n"
)

result = llm.invoke(prompt_template.format(job_description=job_description))
print(result)


job_description = """
SPS-Software Engineer (m/w/d) im Maschinenbau
Glaston Germany GmbH
Neuhausen-Hamberg
Feste Anstellung
Homeoffice möglich, Vollzeit
Erschienen: vor 1 Tag
Glaston Germany GmbH logo
SPS-Software Engineer (m/w/d) im Maschinenbau
Glaston Germany GmbH
slide number 1slide number 2slide number 3
Glaston ist eine internationale Marke mit weltweit führenden Unternehmen, die für zukunftsweisende Maschinen, Anlagen, Systeme und Dienstleistungen in der Bearbeitung von Architektur-, Fahrzeug- und Displayglas steht.

Mit unserer über 50-jährigen Erfahrung am Standort Glaston Germany GmbH in Neuhausen bei Pforzheim verbessern und sichern wir nachhaltig die Produktivität unserer Kunden bei der Fertigung von Architekturglas. Diesen Erfolg verdanken wir unseren motivierten und engagierten Mitarbeitenden und wurden so zu einem der führenden Anbieter von automatisierten und kundenspezifischen Anlagen.

Der Umgang mit Software liegt dir im Blut und du möchtest bei einem Hidden Champion durchstarten?
Dein Faible für Softwarelösungen und dein Herz für unterschiedliche Technologien sind ideale Voraussetzungen, um Maschinen wieder zu alter Stärke zu verhelfen?
Du hast einen ausgeprägten Servicegedanken und Spaß an der Arbeit mit Kunden?

Dann komm zu Glaston! Wir suchen ab sofort für unseren Bereich Service Upgrades Verstärkung!

SPS-SOFTWARE ENGINEER (M/W/D) IM MASCHINENBAU

Als SPS-Software Engineer (m/w/d) im Maschinenbau sind deine Aufgabengebiete:
Ausarbeitung und Weiterentwicklung von Kundenaufträgen und Upgrade-Konzepten
Selbstständige und termingerechte Bearbeitung von Kundenprojekten und Bereitstellung der notwendigen Dokumente
Unterstützung des Inbetriebnahme- und Servicepersonals im Haus und beim Kunden vor Ort
Diese Anforderungen musst du mitbringen:
Qualifizierte technische Ausbildung: Techniker, Studium oder vergleichbare Qualifikation
Mehrjährige Berufserfahrung im Serviceumfeld, idealerweise im Maschinen- und Anlagenbau
Umfangreiche Kenntnisse in verschiedenen SPS-Programmiersprachen (z.B. S7Classic, TIA, Simotion)
Bei uns profitierst du von folgenden Benefits:
Exzellente Rahmenbedingungen (z.B. attraktives Gehaltsmodell, flexible Arbeitszeiten mit Gleitzeit und Homeoffice-Möglichkeiten)
Attraktives Arbeitsumfeld in idyllisch-ländlicher Lage
Umfangreiche Mobilitätsförderung (z.B. Ladestation für Elektroautos)
Wellbeing am Arbeitsplatz
"""

"""Generated output
No, this job description is not suitable for a junior Java developer.\n\nThe key reasons are:\n\n* … (output reduced)

Problem: The output of the LLM is free text, which might be difficult to parse or interpret in subsequent pipeline steps. 

Solution: Add a specific instruction to a prompt?
"""

prompt_template_enum = (
"Given a job description, decide whether it suits a junior Java developer."
"\nJOB DESCRIPTION:\n{job_description}\n\nAnswer only YES or NO."
)

result = llm.invoke(prompt_template_enum.forma(job_description=job_description))
print(result.content)
#NO

#Parse the output of the llm.The EnumOutputParser converts text output into a corresponding Enum instance
from enum import enum
from langchain.output_parsers import EnumOutputParser
from langchain_core.messages import HumanMessage

class IsSuitableJobEnum(Enum):
  YES = "YES",
  NO = "NO"

parser = EnumOutputParser(enum=IsSuitableJobEnum)
assert parser.invoke("NO") == IsSuitableJobEnum.NO
assert parser.invoke("YES\n") == IsSuitableJobEnum.YES
assert parser.invoke(" YES \n") == IsSuitableJobEnum.YES
assert parser.invoke(HumanMessage(content=" YES \n")) == IsSuitableJobEnum.YES

#Create a chain and combine
chain = llm | parser
result = chain.invoke(prompt_template_enum.format(job_description=job_description))
print(result)

#Make chain as part of LangGraph
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END,Graph

class JobApplicationState(TypedDict):
  job_description: str
  is_suitable: IsSuitableJobEnum
  application: str

analyze_chain = llm | parser

def analyze_job_description(state):
  job_description = state["job_description"]
  prompt = prompt_template_enum.format(job_description=job_description)
  result = analyze_chain.invoke(prompt)
  return {"is_sutable":result}

def is_suitable_condition(state:JobApplicationState):
  return state["is_suitable"] == IsSuitableJobEnum.YES

def generate_application(state):
  print("...generating application...")
  return {"application": "some_fake_application", "actions": ["action2"]}


builder = StateGraph(JobApplicationState)
builder.add_node("analyze_job_description", analyze_job_description)
builder.add_node("generate_application", generate_application)
builder.add_edge(START, "analyze_job_description")
builder.add_conditional_edges(
  "analyze_job_description",is_suitable_condition,
  {True: "generate_application", False: END})
)
builder.add_edge("generate_application", END)
graph = builder.compile()

#Look the created graph
from IPython.display import Image,display
display(Image(graph.get_graph().draw_mermaid_png()))

#Execute workflow
response = graph.invoke({"job_description":job_description})
print(response)


