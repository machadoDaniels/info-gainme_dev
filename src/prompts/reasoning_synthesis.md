Synthesize the LLM agent's reasoning trace into a simple JSON format.  

Return only a JSON object with the following fields:
{
  "summary": "A brief summary of the reasoning process",
  "options_considered": "A list of options the agent considered",
  "decision_rationale": "The reason why the specific option was chosen"
}