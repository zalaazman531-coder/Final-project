# AI-Workflow Reflection

## Tools and workflow

This project was built with Claude Code, an AI coding assistant that runs
inside VS Code.

The first step was giving the AI the assignment brief and asking it to compare
the project against it. That check found several missing pieces, and one real
problem: the original prediction target ("developed vs. developing country")
was a bad choice, because the model could not be trained, because the data was circular. Calculations on developed vs developing countries are mostly based on income and GDP indicators so asking the model to predict whether a country is developed or developing would mean the model would basically just repeat data. The target was changed to something fairer — predicting whether a
country's life expectancy is Low, Medium, or High.

From there, the AI wrote the main parts of the project: the new data columns
(for example, merging three vaccination columns into one average), the model
training for both tasks, and the web app. Long training runs were done in the
background while other work continued.

Extra tools (called MCP servers) were also used:

- **Playwright** — opened the web app in a real browser and tested it like a
  user would: filled in the form, pressed Predict, and checked the results
  showed up correctly.
- **GitHub** — used to upload the project to the online repository.
- **Context7** — available for looking up current documentation of the coding
  libraries.

## Checking the AI's work

AI-written code can look right and still be wrong, so nothing was trusted
without a test. To be transparent about who did what: the AI ran the checks
itself, using its own tools, on the developer's instructions — and the
developer verified the outcomes and questioned every decision until it could
be explained in plain words. The checks are the kind whose results cannot be
faked: either the report runs or it stops with an error; either the numbers
agree or they don't.

Four checks were done:

1. **Does it run?** After every change, the AI re-ran the full report from
   the beginning. If any step is broken, the run stops with an error. This
   caught a real bug once: one model could not handle the group names "Low",
   "Medium" and "High" as words, so they were replaced with the numbers 0, 1
   and 2.
2. **Is the test fair?** A model must be graded on countries it has never
   seen — otherwise it is like testing a student on the exact questions they
   studied. The code that separates the learning data from the test data was
   reviewed to make sure nothing from the test data reaches the model during
   learning. This matters because such a mistake shows no error message — the
   scores just come out looking better than they really are.
3. **Are the scores believable?** Two simple questions were asked of every
   result. Does the model beat just always guessing the average? (Yes, all of
   them do, by a lot.) And does it score almost as well on new countries as on
   the ones it learned from? (Yes — if it didn't, it would mean the model
   memorized its learning data instead of finding real patterns.)
4. **Does the app work?** The AI double-checked that the app runs: it opened
   the app in a real browser, filled in the form with average values, and
   pressed Predict. The app answered 71.8 years and the "Medium" group. The
   Medium group covers 67–74 years, so the number lands inside the group —
   the app's two answers agree with each other, which shows it is wired up
   correctly.

The developer's own verification, on top of the AI's checks, was refusing to
accept any step that could not be explained in plain language — including
questioning what the checks themselves proved and who performed them.

## Cost and effort

The project took about one working day. Without AI help it would likely have
taken three to four times longer, especially the web app and the debugging.
Most of the human time went into decisions (what to predict, how to split the
data fairly, what weaknesses to admit), reading the code, and checking the
results — not into typing. Every choice in the project can be explained and
defended without the AI's help.
