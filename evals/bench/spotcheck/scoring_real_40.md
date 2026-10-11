# Task 1 spot check: 40 real answers and the labeler's level

Random 40 of the real soft answers from the six evaluation runs, with the level zai-org/GLM-5.3 gave them (the gold label). Write your own level in the blank and note any disagreement.

## 1. software-engineer / sql_depth (`real:20261009T211706Z:12:sql_depth`)

- Question: Can you tell me about a slow SQL query you tuned, and what you changed?
- Rubric: Names a concrete query problem and a concrete fix (an index, rewriting a join, a window function, reading the query plan).
- Answer: I read the query plan, added a composite index on merchant_id and created_at, and rewrote the join to filter by date range first, which cut the runtime from 40 seconds to under 2.
- Labeler: **fully_meets**: Concrete problem and multiple concrete fixes: read the query plan, added a composite index, rewrote the join to filter by date range first, with before/after timings.
- Yours: fully_meets

## 2. software-engineer / sql_depth (`real:20261009T203518Z:13:sql_depth`)

- Question: Can you tell me about a slow SQL query you tuned, and what you changed?
- Rubric: Names a concrete query problem and a concrete fix (an index, rewriting a join, a window function, reading the query plan).
- Answer: I had a reconciliation query that was slow because it joined a large transactions table to a merchants table on a non-indexed column. I added a composite index on the transactions table for the merchant_id and settlement_date columns, and rewrote the join to filter by settlement_date first, which cut the query time from 12 seconds to under a second.
- Labeler: **fully_meets**: Concrete problem (join on a non-indexed column between transactions and merchants) and concrete fixes (composite index on merchant_id/settlement_date plus rewriting the join order), with measured improvement.
- Yours: fully_meets

## 3. customer-support / frustrated_customer (`real:20261009T192610Z:25:frustrated_customer`)

- Question: Tell me about a time you calmed down an upset customer or friend. What did you do?
- Rubric: Describes listening, acknowledging the frustration, staying polite, and a concrete step that resolved or escalated the issue.
- Answer: I usually listen without interrupting and apologize for the trouble they are facing. Then I calmly walk them through the solution or escalate it if I can't fix it myself.
- Labeler: **partly_meets**: Touches all four rubric elements (listening without interrupting, apologizing, staying calm, walking through a solution or escalating), but only in general habitual terms ('I usually') with no concrete specific incident.
- Yours: partly_meets

## 4. staff-nurse / ventilator_care (`real:20261009T203518Z:18:ventilator_care`)

- Question: Can you describe how you care for a patient on a ventilator during your shift?
- Rubric: Mentions concrete checks such as airway and suction, ventilator settings and alarms, vitals and sedation monitoring, and documentation.
- Answer: I check airway patency and suction as needed, verify ventilator settings and alarms, and monitor vitals and sedation levels. I document all these observations in the patient's record.
- Labeler: **fully_meets**: Concretely covers all rubric elements: airway patency and suction, ventilator settings and alarms, vitals and sedation monitoring, and documentation.
- Yours: fully_meets

## 5. data-analyst / ab_testing (`real:20261009T211706Z:7:ab_testing`)

- Question: How would you decide whether an A/B test result is real and not just noise?
- Rubric: Mentions sample size or power, statistical significance or confidence intervals, and running the test long enough or checking for bias.
- Answer: I check the sample size and power before launch, then use confidence intervals and p-values to decide, and always run the test for the pre-determined duration to avoid peeking bias.
- Labeler: **fully_meets**: Covers all three rubric elements: sample size/power, confidence intervals and p-values, and running the pre-determined duration to avoid peeking bias.
- Yours: fully_meets

## 6. customer-support / frustrated_customer (`real:20261009T211706Z:24:frustrated_customer`)

- Question: Tell me about a time you calmed down an upset customer or friend. What did you do?
- Rubric: Describes listening, acknowledging the frustration, staying polite, and a concrete step that resolved or escalated the issue.
- Answer: I usually listen to them, acknowledge their frustration, and stay polite while checking the details. Then I either resolve the issue myself or escalate it to a senior team if needed.
- Labeler: **partly_meets**: Mentions listening, acknowledging, and staying polite, but the resolution is vague ('resolve it myself or escalate if needed') with no concrete step — general terms only.
- Yours: partyly_meets

## 7. data-analyst / pandas (`real:20261009T211706Z:7:pandas`)

- Question: What have you used Python and pandas for in your work?
- Rubric: Describes real data cleaning or analysis in pandas, such as merges, group-bys, handling missing values, or automating a report.
- Answer: As for Python, I regularly use pandas to merge and group-by large exports from Snowflake, handle missing values, and automate weekly category performance reports that are emailed to managers.
- Labeler: **fully_meets**: Describes merges, group-bys, handling missing values, and automated reporting in pandas.
- Yours: fully_meets

## 8. software-engineer / sql_depth (`real:20261009T211706Z:13:sql_depth`)

- Question: Can you tell me about a slow SQL query you tuned, and what you changed?
- Rubric: Names a concrete query problem and a concrete fix (an index, rewriting a join, a window function, reading the query plan).
- Answer: I had a reconciliation query joining four tables that took 12 seconds, so I analyzed the query plan, added a composite index on merchant_id and created_at, and reduced the execution time to under one second.
- Labeler: **fully_meets**: Names a concrete problem (reconciliation query joining four tables, 12s runtime) and concrete fixes (read the query plan, composite index on merchant_id and created_at) with a measured result.
- Yours: fully_meets

## 9. staff-nurse / ventilator_care (`real:20261009T203518Z:17:ventilator_care`)

- Question: Can you describe how you care for a patient on a ventilator during your shift?
- Rubric: Mentions concrete checks such as airway and suction, ventilator settings and alarms, vitals and sedation monitoring, and documentation.
- Answer: I check the airway and suction, verify ventilator settings and alarms, monitor vitals and sedation, and document everything in the HIS. I also ensure the patient remains comfortable and free from complications during their shift.
- Labeler: **fully_meets**: Covers all rubric elements concretely: airway and suction, ventilator settings and alarms, vitals and sedation monitoring, and documentation in the HIS.
- Yours: fully_meets

## 10. customer-support / frustrated_customer (`real:20261009T203518Z:3:frustrated_customer`)

- Question: Tell me about a time you calmed down an upset customer or friend. What did you do?
- Rubric: Describes listening, acknowledging the frustration, staying polite, and a concrete step that resolved or escalated the issue.
- Answer: I listen without interrupting, acknowledge their frustration, and stay polite. For example, a customer was upset about a delayed router; I apologized, checked the tracking, gave a clear update, and offered a callback if it didn't arrive, which calmed them down.
- Labeler: **fully_meets**: Covers all rubric elements concretely: listens without interrupting, acknowledges frustration, stays polite, and takes concrete steps (checked tracking, gave a clear update, offered a callback) that resolved the situation.
- Yours: fully_meets

## 11. customer-support / frustrated_customer (`real:20261009T203518Z:1:frustrated_customer`)

- Question: Tell me about a time you calmed down an upset customer or friend. What did you do?
- Rubric: Describes listening, acknowledging the frustration, staying polite, and a concrete step that resolved or escalated the issue.
- Answer: I listen without interrupting, acknowledge their frustration, and stay calm. For example, I once helped an angry customer by apologizing for the delay, checking their order status, and offering a replacement, which resolved the issue.
- Labeler: **fully_meets**: Explicitly covers all rubric elements: listening without interrupting, acknowledging frustration, staying calm, plus a concrete example with a resolving step (apologized, checked order status, offered replacement).
- Yours: fully_meets

## 12. warehouse-supervisor / wms_experience (`real:20261009T211706Z:22:wms_experience`)

- Question: Which warehouse management system have you used, and what did you do in it?
- Rubric: Names a WMS and describes hands-on tasks in it, such as receiving, putaway, picking waves or cycle counts.
- Answer: I have used SAP EWM for receiving, putaway, picking waves, and cycle counts.
- Labeler: **fully_meets**: Names SAP EWM and lists specific hands-on tasks: receiving, putaway, picking waves, and cycle counts.
- Yours: fully_meets

## 13. data-analyst / ab_testing (`real:20261009T215614Z:7:ab_testing`)

- Question: How would you decide whether an A/B test result is real and not just noise?
- Rubric: Mentions sample size or power, statistical significance or confidence intervals, and running the test long enough or checking for bias.
- Answer: Honestly, it really depends on the specific context of the experiment, but generally I check sample size and power before launch. Then I compare conversion rates with confidence intervals and run the test for at least two full weeks to avoid novelty effects.
- Labeler: **fully_meets**: Covers sample size and power, confidence intervals, and running the test for at least two full weeks to avoid novelty effects (bias).
- Yours: fully_meets

## 14. customer-support / frustrated_customer (`real:20261009T192610Z:3:frustrated_customer`)

- Question: Tell me about a time you calmed down an upset customer or friend. What did you do?
- Rubric: Describes listening, acknowledging the frustration, staying polite, and a concrete step that resolved or escalated the issue.
- Answer: I listen without interrupting, acknowledge their frustration, and apologize sincerely. Then I calmly walk them through the troubleshooting steps, and if it doesn't resolve, I escalate to a senior with all details noted.
- Labeler: **fully_meets**: Covers all rubric elements: listening without interrupting, acknowledging/apologizing, staying calm, and a concrete resolve-or-escalate path (troubleshooting, escalation with details noted).
- Yours: partly_meets: Did not actually mention any instance

## 15. data-analyst / pandas (`real:20261009T223414Z:6:pandas`)

- Question: What have you used Python and pandas for in your work?
- Rubric: Describes real data cleaning or analysis in pandas, such as merges, group-bys, handling missing values, or automating a report.
- Answer: Yes, I have used pandas for data cleaning and analysis, including merging datasets and handling missing values. It helped me automate some of my recurring reports as well.
- Labeler: **partly_meets**: Names merging, handling missing values, and automating reports, but entirely in general terms with no concrete example or detail of actual pandas work.
- Yours: partly_meets

## 16. warehouse-supervisor / wms_experience (`real:20261009T215614Z:21:wms_experience`)

- Question: Which warehouse management system have you used, and what did you do in it?
- Rubric: Names a WMS and describes hands-on tasks in it, such as receiving, putaway, picking waves or cycle counts.
- Answer: Yes, that is correct; I used SAP EWM daily for receiving, putaway, picking waves, and cycle counts. I also handled stock corrections and inventory reports directly in the system.
- Labeler: **fully_meets**: Names SAP EWM and describes daily hands-on tasks: receiving, putaway, picking waves, cycle counts, plus stock corrections and inventory reports in the system.
- Yours: fully_meets

## 17. customer-support / frustrated_customer (`real:20261009T211706Z:2:frustrated_customer`)

- Question: Tell me about a time you calmed down an upset customer or friend. What did you do?
- Rubric: Describes listening, acknowledging the frustration, staying polite, and a concrete step that resolved or escalated the issue.
- Answer: I usually listen to the customer, acknowledge their frustration, and stay polite while I check the order details. Then I either resolve the issue or escalate it to a senior team if needed.
- Labeler: **fully_meets**: Covers listening, acknowledging frustration, staying polite, and a concrete action (checking order details) followed by resolving or escalating to a senior team as needed.
- Yours: partly_meets: does not actually mention a specific instance

## 18. staff-nurse / ventilator_care (`real:20261009T223414Z:17:ventilator_care`)

- Question: Can you describe how you care for a patient on a ventilator during your shift?
- Rubric: Mentions concrete checks such as airway and suction, ventilator settings and alarms, vitals and sedation monitoring, and documentation.
- Answer: I check the airway and suction as needed, verify ventilator settings and respond to alarms, monitor vitals and sedation, and document everything in the patient's record.
- Labeler: **fully_meets**: Covers all rubric elements concretely: airway and suction, ventilator settings and alarm response, vitals and sedation monitoring, and documentation.
- Yours: fully_meets

## 19. data-analyst / ab_testing (`real:20261009T215614Z:5:ab_testing`)

- Question: How would you decide whether an A/B test result is real and not just noise?
- Rubric: Mentions sample size or power, statistical significance or confidence intervals, and running the test long enough or checking for bias.
- Answer: I check the sample size and power before a test, then use confidence intervals and p-values to decide, and I make sure the test runs long enough to avoid bias.
- Labeler: **fully_meets**: The candidate covers all rubric elements: sample size and power, confidence intervals and p-values for significance, and running the test long enough to avoid bias.
- Yours: fully_meets

## 20. customer-support / frustrated_customer (`real:20261009T203518Z:2:frustrated_customer`)

- Question: Tell me about a time you calmed down an upset customer or friend. What did you do?
- Rubric: Describes listening, acknowledging the frustration, staying polite, and a concrete step that resolved or escalated the issue.
- Answer: I usually let them vent first without interrupting, then I apologize and check their details to fix the issue or explain the next step. If I can't solve it, I escalate to a senior agent.
- Labeler: **fully_meets**: Lets them vent without interrupting (listening), apologizes (acknowledging), and takes a concrete step: check details, fix or explain next step, escalate to a senior agent if needed.
- Yours: partly_meets: Does not mention any example

## 21. data-analyst / pandas (`real:20261009T203518Z:6:pandas`)

- Question: What have you used Python and pandas for in your work?
- Rubric: Describes real data cleaning or analysis in pandas, such as merges, group-bys, handling missing values, or automating a report.
- Answer: I use pandas daily to merge datasets, group by categories, and handle missing values. For example, I automated a weekly sales report that pulls data from Snowflake, cleans it with pandas, and emails the summary to category managers.
- Labeler: **fully_meets**: Concretely covers merges, group-bys, handling missing values, and automating a report with pandas, fully matching the rubric.
- Yours: fully_meets

## 22. sales-executive / target_record (`real:20261009T211706Z:11:target_record`)

- Question: Tell me about your monthly targets in your last role and how you did against them.
- Rubric: Gives a concrete target and their result against it over several months, ideally with numbers.
- Answer: In the last six months, I consistently achieved 115-120% of my monthly target of 12 enrolments, averaging 14 enrolments per month.
- Labeler: **fully_meets**: Gives a concrete monthly target (12 enrolments) and a quantified result over six months (115-120%, averaging 14 per month).
- Yours: fully_meets

## 23. customer-support / frustrated_customer (`real:20261009T211706Z:1:frustrated_customer`)

- Question: Tell me about a time you calmed down an upset customer or friend. What did you do?
- Rubric: Describes listening, acknowledging the frustration, staying polite, and a concrete step that resolved or escalated the issue.
- Answer: I would let the customer finish, apologize for the trouble, and say I understand why they are upset. Then I would check their order details and either fix the issue right away or transfer them to a senior agent with a clear summary.
- Labeler: **fully_meets**: Covers letting the customer finish (listening), apologizing and validating the frustration, and a concrete step: check order details, fix it or transfer to a senior agent with a clear summary (escalation).
- Yours: partly_meets: does not mention clear instance

## 24. customer-support / frustrated_customer (`real:20261009T211706Z:3:frustrated_customer`)

- Question: Tell me about a time you calmed down an upset customer or friend. What did you do?
- Rubric: Describes listening, acknowledging the frustration, staying polite, and a concrete step that resolved or escalated the issue.
- Answer: I listened without interrupting, acknowledged their frustration, and apologized for the trouble before calmly walking them through the fix.
- Labeler: **fully_meets**: Covers all rubric elements: listened without interrupting, acknowledged the frustration, apologized politely, and took a concrete resolving step (calmly walking them through the fix).
- Yours: partly_meets: Does not tell about a time

## 25. software-engineer / sql_depth (`real:20261009T195557Z:12:sql_depth`)

- Question: Can you tell me about a slow SQL query you tuned, and what you changed?
- Rubric: Names a concrete query problem and a concrete fix (an index, rewriting a join, a window function, reading the query plan).
- Answer: I had a reconciliation query joining three large tables that took 40 seconds, so I analyzed the query plan and added a composite index on merchant_id and settlement_date to cut it to under a second.
- Labeler: **fully_meets**: Names a specific problem (three-table reconciliation join taking 40s) and a concrete, verified fix (reading the query plan, adding a composite index on merchant_id and settlement_date, cutting runtime to under a second).
- Yours: fully_meets

## 26. sales-executive / target_record (`real:20261009T195557Z:8:target_record`)

- Question: Tell me about your monthly targets in your last role and how you did against them.
- Rubric: Gives a concrete target and their result against it over several months, ideally with numbers.
- Answer: In my last role, my monthly target was 12 enrolments, and I consistently achieved 110% to 120% of that target over the past six months. This means I averaged between 13 and 14 enrolments per month, exceeding my goals by 15% to 20%.
- Labeler: **fully_meets**: Concrete target (12 enrolments/month) with results over six months (110–120%, i.e., 13–14/month). Numbers given throughout.
- Yours: fully_meets

## 27. data-analyst / pandas (`real:20261009T195557Z:4:pandas`)

- Question: What have you used Python and pandas for in your work?
- Rubric: Describes real data cleaning or analysis in pandas, such as merges, group-bys, handling missing values, or automating a report.
- Answer: I use pandas daily to merge large datasets, handle missing values, and group by dimensions for weekly category reports. I also automate recurring data pulls and transformations with pandas scripts.
- Labeler: **fully_meets**: Concrete daily use: merging large datasets, handling missing values, grouping by dimensions for weekly category reports, and automating recurring pulls/transformations with pandas scripts.
- Yours: fully_meets

## 28. data-analyst / pandas (`real:20261009T195557Z:7:pandas`)

- Question: What have you used Python and pandas for in your work?
- Rubric: Describes real data cleaning or analysis in pandas, such as merges, group-bys, handling missing values, or automating a report.
- Answer: I use pandas daily to handle missing values and group by category to automate those weekly sales and funnel reports that feed into my Power BI dashboards.
- Labeler: **fully_meets**: Hits multiple rubric examples concretely: handling missing values, group-bys, and automating weekly sales/funnel reports.
- Yours: fully_meets

## 29. warehouse-supervisor / wms_experience (`real:20261009T203518Z:22:wms_experience`)

- Question: Which warehouse management system have you used, and what did you do in it?
- Rubric: Names a WMS and describes hands-on tasks in it, such as receiving, putaway, picking waves or cycle counts.
- Answer: Yes, I use SAP EWM daily for those tasks. I handle receiving, putaway, picking waves, and cycle counts with it.
- Labeler: **fully_meets**: Names SAP EWM and describes daily hands-on use for receiving, putaway, picking waves, and cycle counts.
- Yours: fully_meets

## 30. data-analyst / pandas (`real:20261009T211706Z:6:pandas`)

- Question: What have you used Python and pandas for in your work?
- Rubric: Describes real data cleaning or analysis in pandas, such as merges, group-bys, handling missing values, or automating a report.
- Answer: I use pandas to clean and analyze data, like merging tables and grouping metrics for reports. It helps me automate some of my recurring tasks.
- Labeler: **partly_meets**: Mentions merging tables, grouping metrics, and automation, but entirely in general terms with no specific dataset, task, or example of actual data cleaning or analysis.
- Yours: partly_meets

## 31. software-engineer / sql_depth (`real:20261009T223414Z:13:sql_depth`)

- Question: Can you tell me about a slow SQL query you tuned, and what you changed?
- Rubric: Names a concrete query problem and a concrete fix (an index, rewriting a join, a window function, reading the query plan).
- Answer: I read the query plan, saw a sequential scan on the transactions table, and added a composite index on (merchant_id, created_at) which cut the runtime from minutes to seconds.
- Labeler: **fully_meets**: Names a concrete problem (sequential scan on the transactions table) and concrete fixes (reading the query plan, composite index on (merchant_id, created_at)) with measured impact.
- Yours: fully_meets

## 32. sales-executive / target_record (`real:20261009T223414Z:9:target_record`)

- Question: Tell me about your monthly targets in your last role and how you did against them.
- Rubric: Gives a concrete target and their result against it over several months, ideally with numbers.
- Answer: In my current customer support role, I do not have a direct sales target, but during my sales internship, I achieved 110% of my monthly target for two consecutive months.
- Labeler: **partly_meets**: Gives a quantified result (110% of monthly target) over two consecutive months in a sales internship, but never states the actual target value, and two months falls short of the 'several months' the rubric asks for; current role has no sales target.
- Yours: partly_meets

## 33. customer-support / frustrated_customer (`real:20261009T195557Z:3:frustrated_customer`)

- Question: Tell me about a time you calmed down an upset customer or friend. What did you do?
- Rubric: Describes listening, acknowledging the frustration, staying polite, and a concrete step that resolved or escalated the issue.
- Answer: I let the customer vent their frustration, apologized sincerely, and repeated their issue back to confirm I understood. Then, I walked them through one clear step, like rebooting their router, and if that didn't work, I escalated the case with a full summary so they didn't have to repeat themselves.
- Labeler: **fully_meets**: Covers all rubric elements with concrete actions: let the customer vent, apologized and repeated the issue back to confirm understanding, stayed calm, walked them through a specific step (rebooting the router), and escalated with a full summary so the customer didn't repeat themselves.
- Yours: partly_meets: describes their usual procedure instead of an instance

## 34. software-engineer / api_design (`real:20261009T223414Z:12:api_design`)

- Question: Tell me about an API you designed that other teams used.
- Rubric: Describes an API they designed, who used it, and at least one design decision such as versioning, idempotency or error handling.
- Answer: I made it idempotent by accepting a client-generated request ID. I also versioned it from day one with /v1/
- Labeler: **partly_meets**: Gives concrete design decisions (idempotency via client-generated request ID, /v1/ versioning), but never describes what the API was or who used it, so most of the rubric is uncovered.
- Yours: partly_meets: just mentnions some random api without context

## 35. warehouse-supervisor / wms_experience (`real:20261009T211706Z:20:wms_experience`)

- Question: Which warehouse management system have you used, and what did you do in it?
- Rubric: Names a WMS and describes hands-on tasks in it, such as receiving, putaway, picking waves or cycle counts.
- Answer: I use SAP EWM daily to manage receiving, putaway, picking waves, and cycle counts. I also reconcile stock after every shift to maintain our 99.5% inventory accuracy.
- Labeler: **fully_meets**: Names SAP EWM and concretely describes receiving, putaway, picking waves, cycle counts, plus stock reconciliation with a specific accuracy metric.
- Yours: fully_meets

## 36. staff-nurse / ventilator_care (`real:20261009T195557Z:17:ventilator_care`)

- Question: Can you describe how you care for a patient on a ventilator during your shift?
- Rubric: Mentions concrete checks such as airway and suction, ventilator settings and alarms, vitals and sedation monitoring, and documentation.
- Answer: I check airway patency and suction as needed, verify ventilator settings and respond to alarms, monitor vitals and sedation level, and document all observations in the HIS.
- Labeler: **fully_meets**: Covers all rubric elements concretely: airway patency and suction, ventilator settings and alarm response, vitals and sedation monitoring, and documentation in the HIS.
- Yours: fully_meets

## 37. warehouse-supervisor / wms_experience (`real:20261009T211706Z:23:wms_experience`)

- Question: Which warehouse management system have you used, and what did you do in it?
- Rubric: Names a WMS and describes hands-on tasks in it, such as receiving, putaway, picking waves or cycle counts.
- Answer: I have worked extensively on SAP EWM, handling inbound receiving, putaway, creating picking waves, and running cycle counts to keep stock accurate.
- Labeler: **fully_meets**: Names SAP EWM and concretely covers receiving, putaway, picking waves, and cycle counts.
- Yours: fully_meets

## 38. customer-support / frustrated_customer (`real:20261009T223414Z:24:frustrated_customer`)

- Question: Tell me about a time you calmed down an upset customer or friend. What did you do?
- Rubric: Describes listening, acknowledging the frustration, staying polite, and a concrete step that resolved or escalated the issue.
- Answer: I usually listen to them, acknowledge their frustration, and stay polite while working through the issue or escalating if needed. It helps to keep things calm and get to a solution.
- Labeler: **partly_meets**: Touches all rubric elements (listen, acknowledge, stay polite, work through or escalate) but only in habitual, general terms with no concrete example or specific action.
- Yours: partly_meets

## 39. customer-support / frustrated_customer (`real:20261009T215614Z:2:frustrated_customer`)

- Question: Tell me about a time you calmed down an upset customer or friend. What did you do?
- Rubric: Describes listening, acknowledging the frustration, staying polite, and a concrete step that resolved or escalated the issue.
- Answer: I usually listen to the customer, acknowledge their frustration, and stay polite while working through the steps or escalating if needed. It helps to keep things calm and get to a solution quickly.
- Labeler: **partly_meets**: Names listening, acknowledgment, and politeness, but only in abstract terms. The action step ("working through the steps or escalating if needed") is vague, with no concrete example or specific resolution/escalation.
- Yours: partly_meets

## 40. data-analyst / pandas (`real:20261009T223414Z:4:pandas`)

- Question: What have you used Python and pandas for in your work?
- Rubric: Describes real data cleaning or analysis in pandas, such as merges, group-bys, handling missing values, or automating a report.
- Answer: I regularly use pandas to merge clickstream and order data, group by category, and handle missing values, and I've automated a weekly sales report that pulls from Snowflake and emails the summary.
- Labeler: **fully_meets**: Concretely covers merging clickstream and order data, group-bys, missing-value handling, and an automated weekly report.
- Yours: fully_meets
