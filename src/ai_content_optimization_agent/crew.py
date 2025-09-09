from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import BrightDataWebUnlockerTool, BrightDataSearchTool
from .llms.gemini_google_search_llm import GeminiWithGoogleSearch
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators
import os
MODEL = os.getenv("MODEL")

web_unlocker_tool = BrightDataWebUnlockerTool()
serp_search_tool = BrightDataSearchTool()
@CrewBase
class AiContentOptimizationAgent():
    """AiContentOptimizationAgent crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools

    @agent
    def title_scraper_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["title_scraper_agent"],
            tools=[web_unlocker_tool], # <--- Web Unlocker tool integration
            verbose=True,
            llm=MODEL,
        )

    @task
    def scrape_title_task(self) -> Task:
        return Task(
            config=self.tasks_config["scrape_title_task"],
            agent=self.title_scraper_agent(),
            max_retries=3,
        )

    @agent
    def query_fanout_researcher_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["query_fanout_researcher_agent"],
            verbose=True,
            llm=GeminiWithGoogleSearch(MODEL), # <--- Gemini integration with the Google Search tool
        )

    @task
    def google_search_task(self) -> Task:
        return Task(
            config=self.tasks_config["google_search_task"],
            context=[self.scrape_title_task()],
            agent=self.query_fanout_researcher_agent(),
            max_retries=3,
            markdown=True,
            output_file="output/query_fanout.md",
        )
    
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            verbose=True
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'], # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'], # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AiContentOptimizationAgent crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )

    @agent
    def main_query_extractor_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["main_query_extractor_agent"],
            verbose=True,
            llm=MODEL,
        )

    @task
    def main_query_extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config["main_query_extraction_task"],
            context=[self.google_search_task()],
            agent=self.main_query_extractor_agent(),
        )

    @agent
    def ai_overview_retriever_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["ai_overview_retriever_agent"],
            tools=[serp_search_tool], # <--- SERP API tool integration
            verbose=True,
            llm=MODEL,
        )

    @task
    def ai_overview_extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config["ai_overview_extraction_task"],
            context=[self.main_query_extraction_task()],
            agent=self.ai_overview_retriever_agent(),
            max_retries=3,
            markdown=True,
            output_file="output/ai_overview.md",
        )

    @agent
    def query_fanout_summarizer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["query_fanout_summarizer_agent"],
            verbose=True,
            llm=MODEL,
        )

    @task
    def query_fanout_summarization_task(self) -> Task:
        return Task(
            config=self.tasks_config["query_fanout_summarization_task"],
            context=[self.google_search_task()],
            agent=self.query_fanout_summarizer_agent(),
            markdown=True,
            output_file="output/query_fanout_summary.md",
        )

    @agent
    def ai_content_optimizer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["ai_content_optimizer_agent"],
            verbose=True,
            llm=MODEL,
        )

    @task
    def compare_ai_overview_task(self) -> Task:
        return Task(
            config=self.tasks_config["compare_ai_overview_task"],
            context=[self.query_fanout_summarization_task(), self.ai_overview_extraction_task()],
            agent=self.ai_content_optimizer_agent(),
            max_retries=3,
            markdown=True,
            output_file="output/report.md",
        )