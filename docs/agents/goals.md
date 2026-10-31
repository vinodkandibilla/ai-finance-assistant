# Goals: turning a wish into a savings scenario

Maya says, “I want to save $50,000 for college.” Goals asks when she wants to reach it. When she replies “in three years,” the resumed task retains the original target and purpose. The deadline turns an aspiration into a calculable scenario.

## What the agent does

Four deterministic tools calculate required monthly contribution, future value, time to target, and feasibility. The model chooses tools and checks coverage instead of calculating arithmetic itself. Contributions occur at month end; nominal annual percentage divided by 12 compounds monthly. Omitted starting savings and return default to zero. Constant hypothetical returns exclude taxes, fees, inflation, and withdrawals; there is no RAG source or invented citation.

## From simple to hard

| Level | Example question | Expected processing |
| --- | --- | --- |
| Simple | “How much monthly saving reaches $10,000 in two years, starting at zero with 0% return?” | Use the required-contribution calculator and disclose assumptions. |
| Medium | “Starting with $2,000 and adding $300 monthly at an assumed 4% annual return, how long to reach $20,000?” | Use time-to-goal and explain the whole-month hypothetical timeline. |
| Hard | “With $5,000 now and $400 monthly at an assumed 5% annual return, can I reach $30,000 in four years? If not, what monthly saving is required?” | Use feasibility and required-contribution tools to cover both questions and any shortfall. |

These are illustrative supported paths, conditional on available data and evidence; they are not captured live responses.

## When it asks for clarification

Required contribution needs target and horizon; future value needs contribution and horizon; time to target needs target and contribution; feasibility needs all three. Missing or ambiguous essential inputs are asked together. Tools require finite nonnegative values and positive whole-month horizons, with a backend maximum of 100 years. The agent has at most three decision turns total, including clarification turns.

## What it declines or cannot perform

Choosing a guaranteed investment, tax rules, live prices, and account actions are outside projection scope. A changed clarification task outside scope needs a new question. Unsupported cash flows must not be invented.

## When the answer remains unknown

An unreachable target is a valid scenario result. Missing information after bounded attempts can produce insufficient; tool/execution failures produce error. More decision turns cannot be assumed just because the plan is complicated.

## How to read the result

Check horizon, inputs, assumed return, timing, and exclusions. The Goals page’s interactive chart is a local hypothetical calculation, not a backend result or forecast. “Ask AI about this scenario” submits those inputs to the backend for explanation.

Implementation: [agent](../../src/finassist/agents/goal_planning/agent.py) and [tools or validation](../../src/finassist/agents/goal_planning/tools.py). Continue with [the agent journey](README.md) and [UI walkthrough](../ui-guide.md).
