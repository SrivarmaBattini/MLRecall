# SmartRecall: Problem Statement, Application Requirements, and Functional Specification

## 1. Project Overview

**Project name:** SmartRecall  
**Domain:** Machine Learning education, personalized learning, reinforcement learning, and spaced-repetition scheduling.

SmartRecall is an interactive web application that helps students learn Machine Learning (ML) concepts and revise them at appropriate times. It combines topic-wise lessons, quizzes, learner-progress tracking, retention/forgetting estimation, and a reinforcement-learning-based revision scheduler.

The website is the learner-facing application. The reinforcement-learning scheduler is the decision-making component behind the personalized revision recommendations.

## 2. Problem Statement

Students learning Machine Learning often study many concepts but struggle to determine which concepts they should revise, when they should revisit them, and how to allocate limited study time across topics. Some concepts may be remembered well, while others may be misunderstood or forgotten after a period without revision. A fixed revision timetable treats all topics similarly and may not reflect an individual student's performance or learning history.

Traditional learning platforms commonly provide lessons, quizzes, and progress summaries, but their revision schedules may not adapt sufficiently to a student's changing performance and forgetting patterns. Students therefore need a system that uses their learning history to prioritize revision rather than simply displaying a static list of topics.

**SmartRecall aims to develop an interactive, reinforcement-learning-based Machine Learning tutoring platform that provides topic-wise lessons, quizzes, learner-progress tracking, and personalized revision recommendations. The application will use quiz outcomes, elapsed time, and review history to estimate each topic's retention state. A budget-constrained reinforcement-learning agent will select the topics that should be revised each day, subject to a configurable daily review limit, with the objective of improving long-term retention while respecting the student's available study capacity.**

The application should make recommendations understandable by showing which topics were selected and why they are due for revision.

## 3. Main Objectives

1. Provide organized learning materials for multiple Machine Learning topics.
2. Allow students to study a topic and explicitly mark the lesson as read.
3. Assess current understanding through topic-specific quizzes.
4. Record quiz scores, attempts, timestamps, and revision history.
5. Estimate topic-level retention or forgetting risk from observed learner activity.
6. Recommend a daily set of topics under a limited review budget.
7. Use a reinforcement-learning-based scheduling policy to make revision decisions.
8. Track progress over time and show how performance changes after revision.
9. Evaluate the scheduler against simple baselines, such as a fixed schedule or prioritizing the weakest topics.
10. Make recommendations transparent enough for students to understand and act on them.

## 4. Target User and Core User Journey

**Target user:** A student learning Machine Learning through the website.

Typical journey:

1. The student opens the dashboard.
2. The student browses available ML topics.
3. The student opens a topic and studies its lesson.
4. The student selects **Mark as Read** after reading.
5. The student takes the topic quiz.
6. The application records the score and attempt details.
7. The learner model updates its estimate of the student's current retention or forgetting risk.
8. The daily revision scheduler selects up to **K topics** for revision.
9. The student opens **Today's Revision** and revises the recommended topics.
10. The student completes a short review quiz or another defined review activity.
11. The application records the outcome and updates learner history for future recommendations.

**Important:** “Mark as Read” records that the student says they have read the material. It does not prove that they understood or mastered it. Quiz results and subsequent review performance provide additional evidence.

## 5. Application Pages and Required Functionality

### 5.1 Authentication and Student Profile

**Purpose:** Associate learning activity and revision history with the correct student.

Required functionality:
- Provide registration and login if multiple student accounts are supported.
- Maintain a unique student identifier.
- Store topic progress, quiz attempts, review history, and revision preferences.
- Allow logout.
- Prevent one student's learning history from being visible to another student.

**Minimum viable version:** If authentication is outside the initial scope, use a single local/demo profile and document the limitation. Keep the data model capable of associating records with a student.

### 5.2 Home Dashboard

**Purpose:** Give the student a quick overview of learning progress and today's priorities.

Required functionality:
- Show the number of available topics.
- Show topics read, not yet read, and topics with quiz attempts.
- Show recent quiz scores and overall progress.
- Display today's revision count and daily review limit.
- Provide a clear link or button to **Today's Revision**.
- Show a useful empty state for a new student.

Example values (illustrative only):
- Topics studied: 8 of 20
- Quizzes completed: 6
- Topics due for revision today: 4
- Daily revision capacity: 5 topics

The interface must calculate these values from stored data, not hard-code them.

### 5.3 Machine Learning Topic Sidebar / Catalogue

**Purpose:** Let students browse and select concepts to learn.

Required functionality:
- Display available ML topics in a sidebar or catalogue.
- Organize topics into categories where useful.
- Show statuses such as **Not Started**, **Read**, **Quiz Attempted**, **Needs Revision**, or **Recently Reviewed**.
- Let the student select a topic and open its lesson page.
- Provide search or filtering if the catalogue becomes large.
- Distinguish topics never studied from topics due for revision.

Suggested initial topics:
- Introduction to Machine Learning
- Supervised, Unsupervised, and Reinforcement Learning
- Linear Regression
- Logistic Regression
- Loss Functions
- Gradient Descent
- Overfitting and Underfitting
- Regularization
- Decision Trees
- Random Forests
- Support Vector Machines
- K-Nearest Neighbors
- Naive Bayes
- Clustering and K-Means
- Principal Component Analysis
- Neural Networks
- Evaluation Metrics
- Cross-Validation
- Feature Engineering
- Model Selection

The topic list should be configurable. These are starting examples, not a fixed syllabus.

### 5.4 Topic Lesson / Learning Content Page

**Purpose:** Provide real educational content rather than using topic names only as scheduling items.

Each topic should have structured lesson content containing, where appropriate:
- Topic title and short introduction.
- Learning objectives.
- Clear concept explanation.
- Important definitions.
- Mathematical formulas and notation.
- Worked examples.
- Code examples where useful.
- Diagrams or illustrations where they improve understanding.
- Common mistakes or misconceptions.
- A short summary or key takeaways.
- A button to start the topic quiz.

The page should support readable formatting for headings, paragraphs, formulas, code, and images.

**Mark as Read functionality**
- Display a **Mark as Read** button at the end of the lesson or in a clearly visible location.
- Record that the student marked the topic as read.
- Store student ID, topic ID, and timestamp.
- Update the displayed topic status.
- Prevent repeated clicks from creating duplicate completion records unless multiple reading sessions are intentionally tracked.
- Allow the student to revisit the lesson later.

**Interpretation:** Marking a topic as read is a reading-completion event. It must not automatically assign a perfect quiz score, set mastery to 100%, or count as proof of learning.

### 5.5 Topic Quiz and Assessment

**Purpose:** Measure the student's current performance on a topic.

Required functionality:
- Provide a quiz associated with each topic.
- Support multiple-choice questions at minimum.
- Optionally support true/false, short-answer, or coding questions if within scope.
- Display one question at a time or a clear question list.
- Allow answer selection and quiz submission.
- Validate required answers or clearly handle unanswered questions.
- Calculate scores for objectively graded questions.
- Display the score, percentage, correct answers, and explanations after submission.
- Record the attempt in the student's history.
- Allow another attempt without deleting previous attempts.
- Make randomized question selection/order configurable if implemented.

Record for each attempt:
- Student ID.
- Topic ID.
- Quiz/attempt ID.
- Start and submission timestamps.
- Number of questions.
- Number of correct answers.
- Total score and percentage.
- Per-question responses, if needed for analytics.
- Attempt number.

**How quiz results should be used:** Scores are observations of current performance. A low score may indicate weak understanding; a high score may indicate stronger current performance. A single score is not automatically a forgetting rate or half-life. Combine scores with elapsed time and repeated observations to estimate retention or forgetting risk.

### 5.6 Learner Progress and History

**Purpose:** Keep a reliable record of what the student studied, how they performed, and what they revised.

Required functionality:
- Show topic-wise reading status.
- Show the latest quiz score for each topic.
- Show score history across attempts.
- Show the date of the last study, quiz, and revision activity.
- Show topics not yet assessed.
- Show revision history: recommended, opened, completed, skipped, or overdue, as applicable.
- Display progress with simple tables, charts, or status indicators.
- Preserve historical records instead of overwriting earlier attempts.

Useful views:
- Topic progress table.
- Quiz score trend over time.
- Recent activity list.
- Revision completion history.
- Topics with low recent scores.
- Topics with high estimated forgetting risk.

Distinguish **observed facts** (e.g., “quiz score: 60%”) from **model estimates** (e.g., “estimated forgetting risk: high”).

### 5.7 Today's Revision Page

**Purpose:** Show which topics the system recommends revising today.

Required functionality:
- Generate a daily revision list for the current student.
- Select no more than **K topics** per day, where K is the configured daily review budget.
- Show each selected topic, reason for recommendation, last review date, latest quiz score, and priority indicator when available.
- Let the student open the lesson or a focused revision summary.
- Provide a short review quiz or defined review task for each selected topic.
- Let the student mark a task complete after the required review activity.
- Record completed, skipped, and unfinished tasks.
- Update the learner model after the review outcome is observed.
- Show remaining daily capacity.
- Handle cases where fewer than K topics are due or no topics are eligible.
- Avoid recommending the same topic repeatedly within a day unless repeated reviews are an explicit design choice.

**Daily budget rule:** If K = 5, the scheduler may select at most five topics for that day's revision list. It may select fewer when fewer topics are eligible.

Define what the budget means. If every topic takes roughly the same effort, K can be the maximum number of topics per day. If review durations vary, a time budget may be more realistic and should be modelled explicitly.

**Recommendation explanation:** Where supported by the data, show a short reason such as:
- “Your latest quiz score was low.”
- “This topic has not been reviewed for several days.”
- “The model estimates that retention may have declined.”
- “This topic is prioritized under today's review limit.”

Only show reasons supported by stored data or scheduler output.

### 5.8 Focused Revision Experience

**Purpose:** Make a recommendation actionable rather than merely displaying a topic name.

Required functionality:
- Open a selected topic in revision mode.
- Show a concise summary of key concepts, formulas, or examples.
- Optionally show flashcards or a short set of review questions.
- Define a review completion event, such as submitting a short quiz.
- Record review start time, completion time, score, and outcome.
- Return the student to the daily revision list.
- Update the last-reviewed timestamp only when the defined review activity is completed.

Opening a topic or clicking a completion button alone should not be treated as evidence that the student remembered the material. A quiz or other explicit review outcome provides stronger evidence.

### 5.9 Settings and Daily Review Budget

**Purpose:** Keep recommendations within the student's available revision capacity.

Required functionality:
- Define a default daily budget K.
- Allow an administrator or student to change K if this is in scope.
- Validate that K is a positive integer.
- Show the active budget on the dashboard and revision page.
- Enforce the K-topic limit in scheduling logic.
- Store the budget value and effective date if changes need to be tracked.

If the experiment uses a fixed K, configure it at system level and state this in the report.

## 6. Intelligence Behind the Application

The website provides the user experience. The learner model and scheduler provide personalization. These components must be connected to actual learner events.

### 6.1 Learner-event collection

Collect events such as:
- Lesson marked as read.
- Quiz started and submitted.
- Quiz score and per-question outcomes.
- Topic review started and completed.
- Review quiz result.
- Time since the last relevant study or review event.
- Topic skipped or left overdue.

Each event should have a timestamp and be associated with a student and topic.

### 6.2 Topic retention / forgetting estimation

The system needs a defined method to estimate the learner's retention state for each topic. Potential inputs include:
- Latest and previous quiz scores.
- Success/failure of review questions.
- Time since the last completed study or review.
- Number and timing of previous reviews.
- Topic-specific performance history.

The project may use a probabilistic memory model, a half-life estimator, or another justified model. The implementation must specify the chosen model and how its parameters are learned.

**Important modelling distinction:**
- A quiz score measures observed performance at a particular time.
- A forgetting rate or half-life describes how retention changes over time.
- These are related but not identical.
- A single quiz score cannot, by itself, reliably establish a personal topic half-life.
- Repeated observations over time, or a clearly documented initial assumption, are needed to estimate forgetting behaviour.

If there is not enough real student data, start with an initial estimate and update it as new quiz/review outcomes arrive. Document initial assumptions.

### 6.3 Reinforcement-learning scheduler

The scheduler decides which eligible topics to recommend under the daily capacity constraint.

A design specification must define:

**State:** A representation of the student's topic-level learning situation, potentially including estimated retention or forgetting risk, latest quiz score, time since last review, number of past reviews, previous outcomes, and remaining daily budget. Choose the exact representation to fit the implemented algorithm and available data.

**Actions:** Choose which topic or topics to revise, subject to the daily budget. Depending on the algorithm, this can be modelled as selecting a set of topics or making sequential review/skip decisions.

**Reward:** Define a measurable objective connected to learning. For example, reward can be based on review-quiz success or improvement in later retention, with a penalty for consuming review capacity. State and justify the final formula. Do not reward the agent merely for selecting a topic or clicking “Mark as Read”; these events do not prove learning occurred.

**State transition:** After a topic is reviewed, the observed outcome and time since review update the topic's state estimate. Topics not reviewed continue to age with elapsed time. Define how scores/outcomes affect the next state.

**Daily constraint:**

\[
\sum_i a_{i,t} \leq K
\]

where \(a_{i,t}=1\) if topic \(i\) is selected for review on day \(t\), and \(0\) otherwise.

**Policy:** The learned policy determines which topics to prioritize given the current state and remaining capacity.

### 6.4 Relationship between real users and simulation

Use real stored learner activity for personalized recommendations whenever it is available.

A simulator with synthetic data may still be useful for:
- Training an RL agent before enough real learner history exists.
- Testing the scheduler under controlled scenarios.
- Comparing scheduling policies under reproducible conditions.
- Stress-testing budget compliance and edge cases.

Synthetic data must be labelled as simulated, not presented as actual student behaviour or learning outcomes.

If the agent is trained in a simulator and the website uses real learner data at inference time, document how real learner state is mapped into the simulator-trained policy and how model mismatch is handled.

## 7. Recommended System Architecture

1. **Frontend / Web UI**
   - Dashboard
   - Topic sidebar and lesson pages
   - Quiz interface
   - Learner progress pages
   - Today's Revision page
   - Settings

2. **Backend API**
   - Student/profile endpoints
   - Topic and lesson endpoints
   - Quiz submission and scoring endpoints
   - Progress and history endpoints
   - Daily revision recommendation endpoint
   - Revision completion endpoint

3. **Content and Question Bank**
   - ML topics
   - Lesson content
   - Quiz questions and answer keys
   - Explanations and revision summaries

4. **Learner History Database**
   - Student profiles
   - Topic reading events
   - Quiz attempts and responses
   - Review attempts and outcomes
   - Daily recommendations
   - Review completion/skipping records
   - Review-budget settings

5. **Retention / Memory Estimation Module**
   - Builds topic-level learner state
   - Estimates retention or forgetting risk
   - Updates estimates after new observations

6. **RL Scheduling Module**
   - Receives current learner state
   - Selects up to K topics
   - Records recommendations and decision metadata
   - Applies the chosen RL policy

7. **Evaluation and Analytics Module**
   - Calculates retention and quiz metrics
   - Checks budget compliance
   - Compares the learned policy against baselines
   - Produces reproducible experiment results

### High-level data flow

```text
Student
  |
  v
Web Application
  |-----------------------> Topic Lessons
  |                              |
  |                              v
  |                        Mark as Read Event
  |
  |-----------------------> Topic Quiz
  |                              |
  |                              v
  |                      Score + Attempt History
  |
  v
Learner History Database
  |
  v
Retention / Forgetting Estimator
  |
  v
Current Topic-Level Learner State
  |
  v
Budget-Constrained RL Scheduler
  |
  v
Today's Revision Recommendations (at most K topics)
  |
  v
Student Completes Review Activity
  |
  v
Review Outcome Stored -> Learner State Updated
```

If a synthetic-data simulator is retained, represent it as a separate training/testing component rather than the source of the website's real student history.

## 8. Suggested Data Model

These entities are a starting point and may be represented as relational tables or equivalent database collections.

### Student
- `student_id`
- `display_name` or necessary profile fields
- `created_at`
- `daily_review_budget` or a reference to settings

### Topic
- `topic_id`
- `title`
- `category`
- `description`
- `lesson_content`
- `revision_summary`
- `is_published`

### Question
- `question_id`
- `topic_id`
- `question_text`
- `question_type`
- `options` for multiple-choice questions
- `correct_answer`
- `explanation`
- `difficulty`, if used

### ReadingEvent
- `reading_event_id`
- `student_id`
- `topic_id`
- `marked_read_at`

### QuizAttempt
- `attempt_id`
- `student_id`
- `topic_id`
- `started_at`
- `submitted_at`
- `score`
- `max_score`
- `percentage`
- `responses` or related response records

### ReviewTask
- `review_task_id`
- `student_id`
- `topic_id`
- `scheduled_for`
- `priority` or scheduler output metadata
- `status` (recommended, opened, completed, skipped, or overdue)
- `completed_at`

### ReviewAttempt
- `review_attempt_id`
- `review_task_id`
- `student_id`
- `topic_id`
- `started_at`
- `completed_at`
- `score`
- `max_score`
- `outcome`

### TopicLearnerState
- `student_id`
- `topic_id`
- `latest_score`
- `last_studied_at`
- `last_reviewed_at`
- `review_count`
- `estimated_retention` or `forgetting_risk`, if supported by the model
- `estimated_half_life`, only if the chosen estimator produces it
- `updated_at`

### DailyRecommendationRun
- `run_id`
- `student_id`
- `date`
- `budget_k`
- `selected_topic_ids`
- `policy_version`
- `created_at`

**Data integrity:** retain historical quiz and review attempts. Current topic state may be recalculated from history, but must not replace the history itself.

## 9. Key Functional Requirements Checklist

Use this as an implementation acceptance checklist.

### Learning content
- [ ] The application displays multiple real ML topics.
- [ ] Each topic has structured lesson content.
- [ ] Students can open a topic from the sidebar/catalogue.
- [ ] Each lesson has a working **Mark as Read** button.
- [ ] Reading completion is stored with a timestamp.

### Quizzes
- [ ] Each topic has a linked quiz.
- [ ] Students can submit answers.
- [ ] The application calculates and displays the score.
- [ ] Correct answers and explanations are shown where appropriate.
- [ ] Every attempt is saved without overwriting earlier attempts.
- [ ] Topic progress displays latest and historical results.

### Learner tracking
- [ ] Reading and quiz history are linked to the student.
- [ ] Relevant events have timestamps.
- [ ] Revision recommendations and completion outcomes are tracked.
- [ ] Observed results are distinguished from estimated retention values.

### Revision recommendations
- [ ] A daily revision page exists.
- [ ] The system selects no more than K topics per day.
- [ ] The page explains recommendations using supported reasons.
- [ ] The student can open each recommended topic in revision mode.
- [ ] A defined review activity can be completed.
- [ ] Review outcomes are saved and used to update learner state.
- [ ] Skipped and unfinished tasks are recorded.
- [ ] Empty and low-activity cases are handled.

### Reinforcement learning and model logic
- [ ] State, actions, reward, transitions, and constraints are documented.
- [ ] The scheduling policy uses learner-state information.
- [ ] The daily budget is enforced in backend scheduling logic, not only in the UI.
- [ ] The model updates from new quiz/review observations.
- [ ] Synthetic simulation data, if used, is identified as simulated.
- [ ] The system does not treat “Mark as Read” as proof of mastery.
- [ ] The system does not equate one quiz score directly with a half-life.

### Evaluation
- [ ] Budget compliance is measured.
- [ ] Retention or later quiz performance is measured using a defined protocol.
- [ ] The RL scheduler is compared with at least one simple baseline.
- [ ] Results are reproducible and use clearly described data.
- [ ] Limitations, including limited real-user data, are documented.

## 10. Evaluation Plan

A project with an RL scheduler should evaluate the scheduling logic, not only whether the website pages work.

### Application functionality tests
- Marking a lesson as read updates its status and timestamp.
- Quiz scoring is correct for known test cases.
- Historical attempts remain available after another attempt.
- Daily recommendations never exceed K.
- Completing a review stores its outcome.
- Repeated page loads do not create duplicate daily recommendation runs unintentionally.
- Students cannot see another student's history in a multi-user implementation.

### Scheduler evaluation
Compare the RL scheduler against appropriate baselines, such as:
1. **Fixed schedule:** revise topics according to a predefined interval.
2. **Lowest-score first:** prioritize topics with the lowest latest quiz score.
3. **Highest-forgetting-risk first:** prioritize topics with the highest estimated forgetting risk.
4. **Random eligible selection:** select eligible topics randomly under the same budget.

Use baselines that can be implemented fairly with available data. Report:
- Later quiz performance or a clearly defined retention metric.
- Average performance across topics.
- Number of reviews used.
- Daily budget compliance.
- Revision completion rate in a real deployment, if measured.
- Variability across multiple runs or participants, where applicable.

Do not claim the RL method improves retention until a fair evaluation supports that conclusion. If only simulated results are available, label them as simulated rather than real-world evidence.

## 11. Scope and Implementation Priorities

### Phase 1 — Functional learning website
- Build the topic catalogue and lesson pages.
- Add Mark as Read.
- Add quizzes and score calculation.
- Store student progress and attempt history.
- Build dashboard and progress views.

### Phase 2 — Revision workflow
- Add daily review budget K.
- Build the Today's Revision page.
- Add revision summaries or short review quizzes.
- Store recommendations, completion, skipping, and outcomes.

### Phase 3 — Learner model
- Define topic-level learner state.
- Implement the retention/forgetting estimator.
- Update state from quiz and review events.
- Show estimated retention or risk only when the model supports it.

### Phase 4 — Reinforcement-learning scheduler
- Specify state, actions, reward, transitions, and budget constraints.
- Train or configure the RL scheduling policy.
- Connect the policy to current learner-state data.
- Enforce the K-topic constraint in backend scheduling logic.

### Phase 5 — Evaluation and refinement
- Compare with simple baselines.
- Test edge cases and budget compliance.
- Measure retention using a consistent evaluation protocol.
- Document assumptions, limitations, and results.

A simulator can be developed in parallel with the website or retained from an existing implementation to train and evaluate the scheduler before enough real learner data is available.

## 12. Out of Scope Unless Explicitly Added

To keep the project manageable, these are optional:
- Live multiplayer features.
- Social feeds, chat, and leaderboards.
- Payment or subscription functionality.
- AI-generated lessons at runtime.
- Automatic grading of unrestricted essays.
- A separate mobile application.
- Complex teacher/admin analytics.
- Multiple languages.
- Calendar integrations and push notifications.

An administrator/content-editor interface is useful if lesson and question content must be maintained through the website. Otherwise, initial content may be maintained in a structured file or database.

## 13. Important Design Decisions to Finalize

Before implementing the RL component, document these decisions:

1. **What counts as a review?** For example, completing a short topic quiz rather than merely opening a page.
2. **What is K?** A maximum number of topics per day or a total time budget.
3. **How is retention measured?** For example, later quiz accuracy, a calibrated probability of recall, or a defined proxy.
4. **How is forgetting estimated?** Specify the model, inputs, parameter-estimation method, and assumptions.
5. **What is the reward?** Connect it to the learning objective and the cost of review capacity.
6. **How is the agent trained?** Specify whether training uses synthetic simulation, historical learner data, or another method.
7. **How is the policy evaluated?** Define test scenarios, baselines, and metrics before comparing results.
8. **How does the app handle a new student?** Define an initial state before sufficient personal history exists.
9. **What happens when a student skips a recommendation?** Record it and define how it affects future scheduling.
10. **How are topic dependencies handled?** Decide whether prerequisite knowledge influences eligibility or priority.

These decisions should be explicit in the implementation and report, not hidden as unexplained assumptions in the code.

## 14. Final Definition of the Project

SmartRecall is a personalized Machine Learning learning-and-revision web application. It combines structured topic lessons, reading-completion tracking, topic quizzes, learner-history storage, retention/forgetting estimation, and a budget-constrained reinforcement-learning scheduler. The scheduler uses observed student performance and review history to select a limited set of topics for daily revision. The application records review outcomes and uses them to improve future estimates and recommendations.

The main technical contribution is not simply displaying ML topics or storing quiz scores. It is the connection between:

**Learner observations → topic-level retention state → constrained scheduling decision → review outcome → updated learner state**

The system should then evaluate whether the scheduler allocates revision opportunities effectively.

Keep these parts distinct:
- **Website functionality:** lessons, quizzes, dashboard, progress, and revision pages.
- **Learner modelling:** estimating topic-level retention or forgetting risk.
- **Scheduling intelligence:** choosing topics under the daily budget.
- **Evaluation:** testing correctness, budget compliance, and learning-related outcomes against baselines.
