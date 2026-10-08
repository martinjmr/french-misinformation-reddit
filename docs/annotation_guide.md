# Annotation guide: stance of a post towards its claim

Each post was retrieved by the keywords of one fact-checked claim. The annotator reads the claim and the post's opening (title and first 60 words), and the first 250 words when the opening is not enough. The annotator then gives one of four labels, always relative to that claim.

| Label | Code | Definition |
|---|---|---|
| Off-topic | H | The post does not address the claim itself. Sharing keywords or the broad subject is not enough: a post about climate policy is off-topic for "2023 is the hottest year on record". |
| Supports | S | The post asserts the claim, relays it as true, or brings arguments or facts in its favour. A news article that presents the claimed fact or event as real counts as support. |
| Refutes | R | The post contradicts the claim, debunks it, corrects it, or mocks it. |
| Discusses | D | The post addresses the claim without taking a side: a question about it, a neutral report of a debate or of what someone else asserts, mixed or unclear positions. |

Rules:
1. Judge only the claim that retrieved the post, even if the post covers other claims.
2. Judge what the author says, whatever the claim's verdict. The verdict is shown as context only.
3. Irony counts for its intended meaning: mocking a claim is refuting it.
4. The post must be substantially about the claim: its main subject, or a clearly identifiable part where the author takes a position on it. A passing mention without a position is off-topic.
5. A related but different event is off-topic: another law, another year, another figure (a daily temperature record for "2023 is the hottest year on record").
6. Reporting someone else's assertion neutrally is Discusses. Presenting the core event as fact is Supports, even for a claim rated misleading. Refutes requires an explicit contradiction or correction.
7. When the post addresses the claim but the stance cannot be decided, choose Discusses. When unsure whether the post addresses the claim at all, choose Off-topic.

## How the labels were produced

The 3,341 posts were labelled with Claude (an LLM) following this guide, claim by claim, reading every post. The labels have not been compared with a second annotator.
