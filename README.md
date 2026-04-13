# Paclytics

![Animation](animation.gif)

[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/thepollitosd/paclytics)

## Overview

Paclytics (Scout Analytics) is a high-performance desktop application for FIRST Robotics Competition (FRC) match analysis and prediction. It leverages data from The Blue Alliance (TBA) API and applies a sophisticated statistical model to provide deep, actionable insights into team and alliance performance. The application features a sleek, responsive interface built with PyQt6, designed for scouts and strategists who need to make data-driven decisions quickly.

## Features

* **Event Leaderboard:** Ranks all teams at an event based on a composite "Total Impact" score, with a detailed breakdown of their contributions across different phases of the match.
* **Team Schedule Viewer:** Quickly view the full match schedule for any team at a selected event.
* **In-Depth Match Prediction:** Generates a comprehensive pre-match analysis report for any upcoming match, including:

  * A clear "Live Projection" of the winning alliance and the confidence of the prediction.
  * Projected scores and a summary of the strategic edge between alliances.
  * Advanced metrics like Tactical System Strength (TSS), Efficiency (PHI), and High-Impact Probability (HIP).
  * A dynamic "Alliance Pulse" visualization showing predicted active shifts and power distribution.
  * A detailed "Combatant Systems" table breaking down individual robot contributions.
  * AI-generated "Narratives" that explain the strategic outlook for each alliance in plain English.
  * Roster summaries with assigned player roles (e.g., Offensive Carry, Defensive Anchor).

## The Analytics Engine

Paclytics goes beyond simple OPR calculations by using a weighted non-negative least squares (NNLS) regression model. This model attributes team contributions to various aspects of the match, providing a more nuanced view of performance. Match data is weighted to give more recent matches a higher influence on the ratings.

### Data Source

All match data is fetched in real-time from [The Blue Alliance (TBA) API v3](https://www.thebluealliance.com/apidocs).

### Predictive Modeling

The core of Paclytics is a system of equations solved using NNLS to determine individual robot ratings for key performance indicators. These ratings are then used to simulate match outcomes by considering not just individual strength, but also alliance composition and strategic factors like active shifts.

### Core Metrics

* **AFR** (Autonomous Fuel Rating): A team's predicted point contribution during the autonomous period.
* **OSR** (Offensive Shift Rating): A team's ability to score during their "active" shifts in tele-op.
* **DSR** (Defensive Strength Rating): A team's ability to suppress the scoring of the opposing alliance.
* **EFR** (Endgame Fuel Rating): A team's predicted point contribution during the endgame.
* **TSP** (Transition Strength Points): Points generated during the transition phase.
* **SVC** (Shift Volatility Contribution): A measure of a team's consistency across shifts.
* **RS** (Robustness Score): A composite score derived from OSR and SVC, indicating reliable offensive power.
* **TA** (Tower Average): A team's average contribution to tower points per match.
* **TSS** (Tactical System Strength): A high-level strategic score combining efficiency, defensive capability, and autonomous impact.
* **PHI** (Efficiency): An alliance's scoring efficiency, capped and penalized for over-investing in a single objective.
* **HIP** (High-Impact Probability): The probability of an alliance winning the autonomous period, which influences the match tempo.

## Technical Architecture

* **Desktop GUI:** The application is built with **PyQt6**, providing a native desktop experience.
* **Reporting Engine:** All reports (leaderboards, schedules, and match analyses) are dynamically generated as HTML and rendered in a **PyQtWebEngine** view. The styling is achieved using **Tailwind CSS**, which is embedded directly into the HTML for a modern and responsive look.
* **Asynchronous Operations:** To ensure the UI remains responsive, all network requests to the TBA API and heavy computational tasks for the analytics model are run in background threads (**QThread**).

## Getting Started

### Prerequisites

* Python 3.8+
* PIP

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/thepollitosd/paclytics.git
   cd paclytics
   ```

2. Install the required Python packages:

   ```bash
   pip install numpy scipy tbaapiv3client PyQt6 PyQt6-WebEngine
   ```

### Running the Application

Execute the main script to launch the application:

```bash
python scout.py
```

## Building from Source

This project is configured to be packaged into a standalone executable using **PyInstaller**. The `scout.spec` file contains the build configuration.

1. Install PyInstaller:

   ```bash
   pip install pyinstaller
   ```

2. Run the build command from the project's root directory:

   ```bash
   pyinstaller scout.spec
   ```

This will create a `dist/scout` folder containing the executable file.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
