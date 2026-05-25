# Technical Assignment: Clean Architecture & Vision AI Integration
## Position: Mid-Level Full-Stack Python Software Engineer

## 1. Context & Architectural Framework

This assignment evaluates your ability to implement clean, decoupled architectural patterns in Python, specifically building upon the **Layered Architecture / Ports and Adapters (Onion Architecture)** concepts detailed in the book ***Architecture Patterns with Python*** (O'Reilly) by Harry Percival and Bob Gregory.

Before starting, review the architectural foundations demonstrated in the book's official companion repository: [github.com/cosmicpython/code](https://github.com/cosmicpython/code).

### Architectural Separation Guidelines:
Your solution must strictly respect the separation of concerns outlined in the text:
* **Domain Model Layer (`domain/model.py`):** Contains pure business logic, domain entities, and value objects. This layer must remain completely isolated from infrastructure concerns, databases, web frameworks (FastAPI/Flask), or machine learning frameworks (PyTorch/Ultralytics).
* **Service Layer / Use Cases (`service_layer/services.py`):** Coordinates the execution of use cases. It handles flow control, interacts with abstractions/ports, and manipulates domain abstractions without coupling directly to concrete external dependencies.
* **Adapters / Infrastructure Layer (`adapters/`):** Implements concrete interfaces to external systems. This includes web framework entry points (API controllers), external network gateways, or heavy ML models.

---

## 2. Assignment Objective

You are required to implement an API service that ingests user images, passes them through an AI vision model inference execution sequence, and returns structured prediction metadata—all structured according to the Cosmic Python decoupled architecture patterns.

### Core Technical Requirements

#### 1. Entrypoint & Image Ingestion (Web Adapter)
* Expose an HTTP `POST` API endpoint (using **FastAPI** or **Flask**) capable of accepting a raw binary `JPEG` image via a `multipart/form-data` request payload.
* Implement basic structural and format validation at the edge layer before passing the binary payload into the application core.

#### 2. Abstract Port for Machine Learning (Domain/Service Boundary)
* Define an explicit, abstract boundary interface (e.g., an Abstract Base Class named `AbstractInferenceEngine`) to decouple the application core from specific third-party machine learning frameworks.
* Your application service layer should only ever depend on this abstract port interface, ensuring the model engine could be swapped seamlessly without rewriting core use-case flows.

#### 3. Vision Model Infrastructure (Concrete Adapter)
* Implement a concrete class (e.g., `YoloInferenceEngine`) implementing your abstract engine.
* Integrate a pre-trained open-source computer vision model such as an **Ultralytics YOLO** model (e.g., `yolov8n.pt`).
* The adapter must accept raw input image streams, pass them to the underlying neural network, and process the raw output tensors.

#### 4. Structured Domain Modeling
* Define clear, immutable Python `dataclasses` or Value Objects within the Domain Layer to encapsulate the prediction results cleanly (e.g., `InferenceResult`, `BoundingBox`, `DetectionLabel`).
* Map the proprietary machine learning output structure into these domain objects within the adapter layer. The API must return a structured, clean JSON representation of this domain data.

#### 5. Test-Driven Verification (TDD Execution)
* Write an integration/E2E test suite using `pytest` following the book's testing philosophy.
* Ensure unit tests remain lightning fast and deterministic by providing a mocked or test-double version of the `AbstractInferenceEngine`.
* Validate the full web request/response cycle via a separate integration test with a valid sample image.

---

## 3. Human as the Architect: AI Orchestration & Guardrails

The modern full-stack development lifecycle frequently leverages autonomous agents or generative AI tools (e.g., Cursor, GitHub Copilot Workspace, Claude Code). While we encourage the use of these tools to speed up execution, an elite engineer must prevent unstructured **"Vibe Coding"**—where developers blindly accept AI-generated code without checking architectural alignment.

In your repository documentation, you are **required** to include a dedicated section explaining:
1.  **AI Orchestration Strategy:** How you structured prompts or workspace constraints to generate clean, decoupled code.
2.  **Architectural Guardrails:** How you intervened or modified AI suggestions to prevent the AI from taking shortcuts, such as tightly coupling database/ML logic directly inside the API controllers or domain layer.

---

## 4. Submission & Evaluation Process

To submit your assignment for evaluation, follow the execution steps precisely.

### Step 1: Create a Repository
* Set up a new **Public** repository in your personal GitHub account.
* Initialize the repository with a clean, structured `README.md` file that includes:
    * A high-level directory structure tree showing where the layers break down.
    * Explicit step-by-step installation instructions for virtual environments, package management tool configurations, and model weights downloads.
    * The explicit command-line steps required to execute the test suite (e.g., `pytest`).

### Step 2: Prepare for Email Submission
* Double-check that your repository is fully accessible to public, unauthenticated users.
* Draft an email to the engineering evaluation team.

### Step 3: Submit Your Work
Send an email using the following parameters:

* **To:** `service@smartsurgerytek.com`
* **Subject Line:** `Mid-Level Machine Learning Engineer Technical Assignment Submission`
* **Email Body Content:**
    * Your full legal name and preferred phone contact details.
    * A brief cover letter introducing yourself, highlighting key architectural takeaways from your implementation.
    * The direct, clickable HTTPS link to your GitHub repository.

### Step 4: Follow Up
* The engineering team will review your implementation for architectural separation, testing rigor, and code elegance.
* If you do not receive an acknowledgment within **one week**, you may follow up with a professional reply on the same email thread.
* Be prepared to walk through your code decisions, explain your abstractions, and discuss your AI orchestration patterns in an upcoming live code review session.

## 5. Bonus
* Create a React web UI to submit the JPEG file by calling the new API endpoint.
