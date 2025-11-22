"""
agent.py

High-level BahrainStatsAgent that:
- Holds a DataRepository
- Uses NLU router to understand questions
- Uses describe_layer to generate human-readable answers
- (NEW) Can also return a JSON-serializable dict with structured data
"""

from typing import Optional, Dict, Any

from .data_layer import load_all_data, DataRepository
from .describe_layer import (
    describe_labour_market,
    describe_governorates,
    describe_housing_units,
    describe_students,
    describe_teachers,
    describe_higher_education,
)
from .nlu_router import classify_intent, extract_year
from .query_layer import (
    get_workers_by_year,
    get_top_occupations,
    get_households_by_governorate,
    get_latest_density,
    get_housing_units_summary,
    get_students_summary,
    get_teachers_summary,
    get_higher_education_summary,
)


class BahrainStatsAgent:
    """
    Main public interface for the Bahrain Statistical AI Agent.

    Typical usage:
        agent = BahrainStatsAgent(data_path="data/bahrain_master")
        text_answer = agent.answer_question("Summarize the labour market in 2022")
        json_answer = agent.answer_question_json("Summarize the labour market in 2022")
    """

    def __init__(self, data_path: str = "data/bahrain_master"):
        self.data_path = data_path
        self.repo: DataRepository = load_all_data(data_path)

    # Optional helper if you ever want to reload the data without restarting Python
    def refresh_data(self):
        """Reload CSVs from disk into the DataRepository."""
        self.repo = load_all_data(self.data_path)

    def _get_default_year(self) -> Optional[int]:
        """
        Try to infer a sensible default year from labour_master;
        if not available, from any other dataset that has a 'year' column.
        """
        for df in [
            self.repo.labour_master,
            self.repo.occupation_workers,
            self.repo.households,
            self.repo.population_density,
            self.repo.housing_units,
            self.repo.students,
            self.repo.teachers,
            self.repo.higher_education,
        ]:
            if not df.empty and "year" in df.columns:
                try:
                    return int(df["year"].max())
                except Exception:
                    continue
        return None

    # ---------- TEXT ANSWER (what you already had) ----------

    def answer_question(self, question: str) -> str:
        """
        Main method: takes a natural language question and returns a textual answer.
        """
        intent = classify_intent(question)
        default_year = self._get_default_year()
        year = extract_year(question, default_year)

        # Route based on intent
        if intent == "labour_overview":
            return describe_labour_market(self.repo, year)
        elif intent == "top_occupations":
            return describe_labour_market(self.repo, year)  # could be specialised later
        elif intent == "households":
            return describe_governorates(self.repo)
        elif intent == "density":
            return describe_governorates(self.repo)
        elif intent == "housing_units":
            return describe_housing_units(self.repo)
        elif intent == "students":
            return describe_students(self.repo)
        elif intent == "teachers":
            return describe_teachers(self.repo)
        elif intent == "higher_education":
            return describe_higher_education(self.repo)
        else:
            # Fallback generic help
            help_text = [
                "I didn't fully understand that question, but here is what I can do:",
                "",
                "- Summarize the labour market in a given year",
                "- Show most common occupations",
                "- Describe households and population density by governorate",
                "- Summarize housing units",
                "- Give an overview of students and teachers",
                "- Describe higher education students",
                "",
                "Try asking, for example:",
                "- 'Summarize the labour market in 2022'",
                "- 'What are the most common occupations?'",
                "- 'Describe households and population density by governorate'",
                "- 'Give an overview of higher education students'",
            ]
            return "\n".join(help_text)

    # ---------- JSON ANSWER (NEW) ----------

    def answer_question_json(self, question: str) -> Dict[str, Any]:
        """
        Returns a JSON-serializable dict containing:
        - the question
        - detected intent
        - detected/assumed year
        - human-readable answer_text
        - structured data (tables) when available

        This is what you can return to the client as JSON.
        """
        intent = classify_intent(question)
        default_year = self._get_default_year()
        year = extract_year(question, default_year)

        # Reuse the existing text answer
        answer_text = self.answer_question(question)

        payload: Dict[str, Any] = {
            "question": question,
            "intent": intent,
            "year": year,
            "data_source_path": self.data_path,
            "answer_text": answer_text,
            "data": None,          # will fill below
        }

        try:
            # Attach structured data depending on intent
            if intent in ("labour_overview", "top_occupations"):
                workers = get_workers_by_year(self.repo, year)
                top_occ = get_top_occupations(self.repo, year, top_n=10)

                payload["data"] = {
                    "workers_by_nationality": (
                        workers.to_dict(orient="records") if not workers.empty else []
                    ),
                    "top_occupations": (
                        top_occ.to_dict(orient="records") if not top_occ.empty else []
                    ),
                }

            elif intent in ("households", "density"):
                households = get_households_by_governorate(self.repo)
                density = get_latest_density(self.repo)

                payload["data"] = {
                    "households_by_governorate": (
                        households.to_dict(orient="records") if not households.empty else []
                    ),
                    "population_density": (
                        density.to_dict(orient="records") if not density.empty else []
                    ),
                }

            elif intent == "housing_units":
                units = get_housing_units_summary(self.repo)
                payload["data"] = {
                    "housing_units_summary": (
                        units.to_dict(orient="records") if not units.empty else []
                    )
                }

            elif intent == "students":
                students = get_students_summary(self.repo)
                payload["data"] = {
                    "students_summary": (
                        students.to_dict(orient="records") if not students.empty else []
                    )
                }

            elif intent == "teachers":
                teachers = get_teachers_summary(self.repo)
                payload["data"] = {
                    "teachers_summary": (
                        teachers.to_dict(orient="records") if not teachers.empty else []
                    )
                }

            elif intent == "higher_education":
                he = get_higher_education_summary(self.repo)
                payload["data"] = {
                    "higher_education_summary": (
                        he.to_dict(orient="records") if not he.empty else []
                    )
                }

            else:
                # Unknown intent: no structured data
                payload["data"] = None

        except Exception as e:
            # In case anything goes wrong, don't crash – just record the error
            payload["data"] = None
            payload["data_error"] = str(e)

        return payload
