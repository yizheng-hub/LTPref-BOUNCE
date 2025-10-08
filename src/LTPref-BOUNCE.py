# -*- coding: utf-8 -*-
"""
This script runs a simulation to infer a user's personalized thermal comfort
preference through a multi-turn dialogue.

The core workflow is as follows:
1.  Load preprocessed thermal comfort data.
2.  Initialize a System Agent (LLM, responsible for asking questions) and
    multiple User Agents (LLM, simulating different user behavior types).
3.  For each selected user from the dataset, simulate dialogues for each of the
    pre-defined user behavior types: "precise", "fuzzy", and "noisy".
4.  Within each dialogue turn:
    a. The System Agent selects an optimal PMV point to query based on the
       current Bayesian belief distribution.
    b. The System Agent, using a ReAct loop, generates a natural language
       question and a contextually appropriate scenario.
    c. The User Agent generates a response based on its predefined behavior
       and its ground-truth comfort preference. This response is then
       re-validated to ensure experimental control.
    d. A Natural Language Inference (NLI) model analyzes the user's response
       to derive a soft label (omega) representing preference strength.
    e. The Bayesian belief distribution over the user's preference is updated
       using the soft label.
5.  The dialogue concludes when the inferred PMV range converges or the
    maximum number of turns is reached.
6.  Detailed history, final inferred results, and probability distribution
    plots are saved for each simulation to the 'output' folder.
7.  After all simulations, results are aggregated from the JSON files in the
    'output' folder and a final summary report is generated.
"""

import pandas as pd
import numpy as np
from pythermalcomfort.models import pmv_ppd_ashrae
import random
import re
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import json
import logging
import os
import datetime
import torch # Added import
import gc # Added import for garbage collection

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Assume your Agent class is in agent.py
from agent import Agent
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- Set CUDA/CPU device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device for NLI model: {device}")


# --- Helper for keyword matching with word boundaries ---
def contains_word_match(text: str, keywords: list[str]) -> bool:
    """
    Checks if the text contains any of the keywords as whole words.
    Uses regex word boundaries for precise matching.
    """
    text_lower = text.strip().lower()
    for kw in keywords:
        # \b ensures whole word match (e.g., "comfort" won't match "uncomfortable")
        if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower):
            return True
    return False

# --- Global Configuration Parameters ---
# File Paths based on the new folder structure
PROCESSED_DATA_INPUT_PATH = "../data/CTCD_processed_thermal_comfort_data.csv"
# or use ASHRAE_cleaned_OFFICE_for_simulation.csv

# NLI Model Paths
NLI_DEBERTA_ENHANCED_PATH = '../models/deberta-v3-large-mnli-fever-anli'

# PMV Calculation Constants
ASSUMED_RH_PERCENT = 50       # % Assumed relative humidity for inverse PMV calculation
ASSUMED_AIR_VELOCITY_MPS = 0.1 # m/s Assumed air velocity for inverse PMV calculation

# PMV Range for calculations
PMV_MIN_VALUE = -3.0
PMV_MAX_VALUE = 3.0
PMV_STEP_SIZE = 0.1

# Simulation Parameters
MAX_DIALOGUE_ROUNDS = 7
CONVERGENCE_THRESHOLD_PMV = 0.8 # Dialogue ends if inferred PMV range width is <= this value
INFERENCE_CONFIDENCE_LEVEL = 0.8 # Confidence level for extracting PMV range from distribution
INFERENCE_CONFIDENCE_LEVEL_DYNAMIC = 0.6

# Bayesian Update Parameters
BAYESIAN_TAU = 0.5           # Sensitivity of the Gaussian likelihood model
BAYESIAN_LAMBDA_THRESHOLD = 0.5 # Confidence threshold for NLI to avoid neutral update

# LLM Agent Configuration Paths
USER_AGENT_CONFIG_PATHS = {
    "precise": "../configs/user_agent_precise.json",
    "fuzzy": "../configs/user_agent_fuzzy.json",
    "noisy": "../configs/user_agent_noisy.json"
}
SYSTEM_AGENT_CONFIG_PATH = "../configs/system_agent.json"

# --- NEW: User Behavior Simulation Parameters ---
NOISE_PROBABILITY = 0.1 # Probability for noisy user to flip or change to 'not sure'
USER_BEHAVIOR_TYPES = ["precise", "fuzzy", "noisy"] # List of user behavior patterns

# --- NEW: Output Management ---
OUTPUT_FOLDER_NAME = "../output/full_simulation_results_output" # Define the output folder name

# --- Data Loading and Preprocessing (now simplified to direct load) ---

# Define common fuzzy phrases for forcing
FUZZY_POSITIVE_PHRASES = [
    "Yes, that sounds comfortable.",
    "Yes, it feels just right.",
    "Yes, I would definitely prefer that.",
    "Yes, I enjoy that feeling.",
    "Yes, this is perfect."
]
FUZZY_NEUTRAL_PHRASES = [
    "Not sure, I have mixed feelings.",
    "Not sure, it's a bit in-between for me.",
    "Not sure, it's hard to decide.",
    "I'm not entirely sure, to be honest.",
    "It's neither good nor bad."
]
FUZZY_NEGATIVE_PHRASES = [
    "No, that feels uncomfortable.",
    "No, I definitely would not prefer that.",
    "No, it's far too extreme for me.",
    "No, I dislike this sensation.",
    "No, this is not comfortable at all."
]

# Define keywords for fuzzy users
FUZZY_POSITIVE_KEYWORDS = ["yes", "comfortable", "just right", "pleasant", "fine", "good"]
FUZZY_NEUTRAL_KEYWORDS = ["not sure", "mixed feelings", "in-between", "hard to decide", "don't know", "unsure"]
FUZZY_NEGATIVE_KEYWORDS = ["no", "uncomfortable", "too hot", "too cold", "not prefer", "dislike", "hate", "bad", "extreme", "way too cold", "way too hot", "not for me", "can't stand", "freezing", "sweltering"]

# --- NLI Model Initialization ---
NLI_MODELS = {
    'deberta_enhanced': {
        'model': AutoModelForSequenceClassification.from_pretrained(NLI_DEBERTA_ENHANCED_PATH).to(device),
        'tokenizer': AutoTokenizer.from_pretrained(NLI_DEBERTA_ENHANCED_PATH)
    }
}

def nli_soft_label(question: str, response: str, model_type: str = 'deberta_enhanced', user_behavior_type: str = 'precise') -> tuple[float, float]:
    """
    Uses an NLI model to infer the entailment/contradiction/neutral relationship
    between a question (premise) and a response (hypothesis) to get a soft label.
    For 'precise' user behavior, it applies strict "Yes/No/Not sure" mapping.
    For other user behaviors, it relies on the NLI model for semantic interpretation.

    Args:
        question (str): The premise (question asked by the system).
        response (str): The hypothesis (user's response).
        model_type (str): The type of NLI model to use.
        user_behavior_type (str): The type of user behavior ('precise', 'fuzzy', 'noisy').

    Returns:
        tuple[float, float]: A tuple containing:
            - omega (float): A score representing preference strength (0.0 for contradiction, 1.0 for entailment, 0.5 for neutral).
            - confidence (float): The maximum probability among entailment, neutral, and contradiction, representing the model's confidence.
    """
    response_lower = response.strip().lower()

    # For 'precise' users, use hard-coded mapping for strict control
    if user_behavior_type in ["precise", "noisy"]:
        if response_lower in ["yes", "yes.", "yes!"]:
            return 1.0, 1.0
        elif response_lower in ["not sure", "not sure."]:
            return 0.5, 1.0
        elif response_lower in ["no", "no.", "no!"]:
            return 0.0, 1.0

        logging.warning(f"User type '{user_behavior_type}' returned unexpected response: '{response}'. Falling back to NLI.")

    # For 'fuzzy' users, or precise/noisy users with unexpected responses, use NLI model
    if model_type not in NLI_MODELS:
        logging.error(f"NLI model type '{model_type}' not found. Defaulting to DeBERTa.")
        model_type = 'deberta_enhanced' # Fallback

    model = NLI_MODELS[model_type]['model']
    tokenizer = NLI_MODELS[model_type]['tokenizer']

    id2label = model.config.id2label
    label2id = {v: k for k, v in id2label.items()}

    premise = question.strip()
    hypothesis = response.strip()

    # Modification: Ensure inputs are also moved to the specified device (CPU or CUDA)
    inputs = tokenizer.encode(premise, hypothesis, return_tensors='pt', max_length=512, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(inputs)
    logits = outputs.logits
    probs = logits.softmax(dim=1)[0]  # [contradiction, neutral, entailment]

    # Use dynamic mapping to get the probability for each label
    try:
        contra_score = probs[label2id['contradiction']].item()
        neutral_score = probs[label2id['neutral']].item()
        entail_score = probs[label2id['entailment']].item()
    except KeyError as e:
        logging.error(f"Error: NLI model's labels do not contain expected 'contradiction', 'neutral', 'entailment'. Found labels: {id2label}. Error: {e}")
        return 0.5, 0.0 # Return neutral with zero confidence as a fallback

    omega = entail_score * 1.0 + neutral_score * 0.5 + contra_score * 0.0
    confidence = max(entail_score, neutral_score, contra_score)

    # --- NEW: Fuzzy User NLI Re-validation Mechanism ---
    if user_behavior_type == "fuzzy":
        is_strong_positive_by_keywords = contains_word_match(response, FUZZY_POSITIVE_KEYWORDS) and \
                                         not contains_word_match(response, FUZZY_NEGATIVE_KEYWORDS) and \
                                         not contains_word_match(response, FUZZY_NEUTRAL_KEYWORDS)

        is_strong_negative_by_keywords = contains_word_match(response, FUZZY_NEGATIVE_KEYWORDS) and \
                                         not contains_word_match(response, FUZZY_POSITIVE_KEYWORDS) and \
                                         not contains_word_match(response, FUZZY_NEUTRAL_KEYWORDS)

        is_strong_neutral_by_keywords = contains_word_match(response, FUZZY_NEUTRAL_KEYWORDS) and \
                                        not is_strong_positive_by_keywords and \
                                        not is_strong_negative_by_keywords

        # Re-validate positive cases
        if is_strong_positive_by_keywords and omega < 0.7:
            original_omega = omega
            omega = 0.9 # Force to high positive
            logging.info(f"Fuzzy NLI Re-validation: Forced omega from {original_omega:.2f} to {omega:.2f} (Strong positive keywords found, original NLI too low). Response: '{response}'")

        # Re-validate negative cases
        elif is_strong_negative_by_keywords and (omega > 0.3 or confidence < 0.7):
            original_omega = omega
            omega = 0.1 # Force to high negative
            logging.info(f"Fuzzy NLI Re-validation: Forced omega from {original_omega:.2f} to {omega:.2f} (Strong negative keywords found, original NLI too high/low confidence). Response: '{response}'")

        # Re-validate neutral cases
        elif is_strong_neutral_by_keywords and (omega < 0.4 or omega > 0.6):
            original_omega = omega
            omega = 0.5 # Force to neutral
            logging.info(f"Fuzzy NLI Re-validation: Forced omega from {original_omega:.2f} to {omega:.2f} (Strong neutral keywords found, original NLI not neutral). Response: '{response}'")

    return omega, confidence

# --- PMV Calculation Functions ---
def calculate_pmv(air_temp: float, mrt: float, rh: float, air_vel: float, clo: float, met: float) -> float | None:
    """
    Calculates the Predicted Mean Vote (PMV) using the ASHRAE standard.

    Args:
        air_temp (float): Air temperature in Celsius (°C).
        mrt (float): Mean radiant temperature in Celsius (°C).
        rh (float): Relative humidity in percentage (%).
        air_vel (float): Air velocity in meters per second (m/s).
        clo (float): Clothing insulation in clo units.
        met (float): Metabolic rate in met units.

    Returns:
        float | None: The calculated PMV value, or None if calculation fails.
    """
    try:
        result = pmv_ppd_ashrae(
            tdb=air_temp,
            tr=mrt,
            vr=air_vel,
            rh=rh,
            clo=clo,
            met=met
        )
        return result.pmv
    except Exception as e:
        logging.error(f"PMV calculation failed for inputs: air_temp={air_temp}, mrt={mrt}, rh={rh}, air_vel={air_vel}, clo={clo}, met={met}. Error: {e}")
        return None

def find_temp_for_pmv(target_pmv: float, clo: float, met: float, rh: float = ASSUMED_RH_PERCENT, air_vel: float = ASSUMED_AIR_VELOCITY_MPS, initial_guess: float = 23.0) -> float | None:
    """
    Finds the air temperature (assuming mrt=air_temp) that yields a target PMV
    for given clothing and metabolic rates. Uses brentq for root finding, with
    a fallback to linear search if brentq preconditions are not met.

    Args:
        target_pmv (float): The desired PMV value.
        clo (float): Clothing insulation in clo units.
        met (float): Metabolic rate in met units.
        rh (float): Relative humidity in percentage (%).
        air_vel (float): Air velocity in meters per second (m/s).
        initial_guess (float): Initial guess for the temperature (not directly used by brentq, but for context).

    Returns:
        float | None: The air temperature in Celsius (°C) that results in the target PMV,
                      or None if a suitable temperature cannot be found.
    """
    def pmv_difference(temp: float) -> float:
        """Calculates the difference between calculated PMV and target PMV."""
        calculated_pmv = calculate_pmv(air_temp=temp, mrt=temp, rh=rh, air_vel=air_vel, clo=clo, met=met)
        if calculated_pmv is None:
            return 999999 * np.sign(target_pmv - 0.0) if target_pmv != 0.0 else 999999
        return calculated_pmv - target_pmv

    lower_bound = 10.0
    upper_bound = 40.0

    try:
        diff_lower = pmv_difference(lower_bound)
        diff_upper = pmv_difference(upper_bound)

        if np.sign(diff_lower) == np.sign(diff_upper):
            logging.warning(f"PMV target {target_pmv} might be outside achievable range for bounds [{lower_bound}, {upper_bound}]. Falling back to linear search.")
            temp_values = np.linspace(lower_bound, upper_bound, 1000)
            pmv_diffs = [pmv_difference(t) for t in temp_values]
            min_diff_idx = np.argmin(np.abs(pmv_diffs))
            return temp_values[min_diff_idx]
        
        result_temp = brentq(pmv_difference, lower_bound, upper_bound)
        return result_temp
    except ValueError as e:
        logging.error(f"Brentq failed to find a root for target PMV {target_pmv}. Error: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred in find_temp_for_pmv: {e}")
        return None


def convert_pmv_range_to_temp_range(pmv_min: float, pmv_max: float, clo: float, met: float) -> tuple[float | None, float | None]:
    """
    Converts a PMV range to a corresponding temperature range.

    Args:
        pmv_min (float): The minimum PMV value.
        pmv_max (float): The maximum PMV value.
        clo (float): Clothing insulation in clo units.
        met (float): Metabolic rate in met units.

    Returns:
        tuple[float | None, float | None]: A tuple (min_temp, max_temp),
                                          or (None, None) if conversion fails.
    """
    temp_for_pmv_min = find_temp_for_pmv(pmv_min, clo, met)
    temp_for_pmv_max = find_temp_for_pmv(pmv_max, clo, met)

    if temp_for_pmv_min is None or temp_for_pmv_max is None:
        return None, None
    
    return min(temp_for_pmv_min, temp_for_pmv_max), max(temp_for_pmv_min, temp_for_pmv_max)

# --- PMV to Natural Language Feeling Mapping ---
PMV_FEELING_DESCRIPTIONS = {
    -3.0: "very cold, biting, unbearable", -2.5: "extremely cold", -2.0: "very cold, uncomfortable",
    -1.5: "quite cold", -1.0: "a bit cold, cool", -0.5: "slightly cool",
    0.0: "neutral, very comfortable, neither hot nor cold",
    0.5: "slightly warm", 1.0: "a bit warm, tending hot", 1.5: "quite warm",
    2.0: "very hot, uncomfortable", 2.5: "extremely hot", 3.0: "very hot, unbearable"
}

def get_pmv_feeling_description(pmv_value: float) -> str:
    """
    Returns a natural language description of the thermal sensation
    closest to the given PMV value.

    Args:
        pmv_value (float): The PMV value.

    Returns:
        str: A natural language description of the thermal sensation.
    """
    closest_pmv_key = min(PMV_FEELING_DESCRIPTIONS.keys(), key=lambda k: abs(k - pmv_value))
    return PMV_FEELING_DESCRIPTIONS[closest_pmv_key]

def map_pmv_point_to_range(pmv_point: float) -> list[float]:
    """
    Maps a single PMV value to a standard PMV range for question generation.

    Args:
        pmv_point (float): A single PMV value.

    Returns:
        list[float]: A list [min_pmv, max_pmv] representing the PMV range.
    """
    if pmv_point <= -2.5:
        return [-3.0, -2.0]
    elif pmv_point <= -1.5:
        return [-2.0, -1.0]
    elif pmv_point <= -0.5:
        return [-1.0, 0.0]
    elif pmv_point <= 0.5:
        return [-0.5, 0.5]
    elif pmv_point <= 1.5:
        return [0.0, 1.0]
    elif pmv_point <= 2.5:
        return [1.0, 2.0]
    else:
        return [2.0, 3.0]

# --- LLM Response Parsing Utilities ---
def extract_json_block(raw_response: str) -> dict | None:
    """
    Extracts a JSON block from a raw LLM response string, handling common LLM output issues
    like surrounding text and internal comments.

    Args:
        raw_response (str): The raw string response from the LLM.

    Returns:
        dict | None: The parsed JSON object, or None if no valid JSON block is found.
    """
    # First, try to find blocks delimited by ```json ... ```
    match = re.search(r"```json\s*(\{.*?\})\s*```", raw_response, re.DOTALL)
    json_candidate = None
    if match:
        json_candidate = match.group(1)
    else:
        # If delimited block fails or is not found, try to find a standalone JSON object
        # This regex searches for the first '{' and the last '}'.
        start_idx = raw_response.find('{')
        end_idx = raw_response.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_candidate = raw_response[start_idx : end_idx + 1]

    if json_candidate:
        # Step 1: Remove C++/JavaScript style comments (// and /* */)
        # Remove single-line comments // ...
        cleaned_json_candidate = re.sub(r"//.*$", "", json_candidate, flags=re.MULTILINE)
        # Remove multi-line comments /* ... */
        cleaned_json_candidate = re.sub(r"/\*.*?\*/", "", cleaned_json_candidate, flags=re.DOTALL)
        
        try:
            return json.loads(cleaned_json_candidate)
        except json.JSONDecodeError as e:
            logging.warning(f"⚠️ JSON parsing failed after cleaning: {e}. Attempted raw JSON candidate: {cleaned_json_candidate.strip()[:200]}...")
    
    logging.warning("⚠️ No valid JSON block or raw JSON string found in LLM response.")
    return None

def extract_question_text(parsed_data: dict) -> str:
    """
    Extracts the 'question' field from a parsed LLM JSON response dictionary.

    Args:
        parsed_data (dict): The parsed JSON dictionary from the LLM.

    Returns:
        str: The extracted question text, or a placeholder if extraction fails.
    """
    if parsed_data and "question" in parsed_data:
        return parsed_data["question"]
    else:
        logging.warning("⚠️ Failed to extract question text from LLM response data.")
        return "(Question extraction failed)"

def extract_scenario(parsed_data: dict) -> dict:
    """
    Extracts the 'scenario' field from a parsed LLM JSON response dictionary.

    Args:
        parsed_data (dict): The parsed JSON dictionary from the LLM.

    Returns:
        dict: The extracted scenario dictionary, or a default scenario if extraction fails.
    """
    if parsed_data and "scenario" in parsed_data:
        return parsed_data["scenario"]
    else:
        logging.warning("⚠️ Failed to extract scenario from LLM response data. Using default scenario.")
        return {
            "air_temp": 24.0,
            "mrt": 24.0,
            "humidity": 50,
            "air_vel": 0.1
        }

# --- Loading System Agent Few-Shot Examples and User Prompt Templates from files ---
try:
    with open("../prompts/system_agent_few_shots.json", 'r', encoding='utf-8') as f:
        SYSTEM_AGENT_FEW_SHOT_EXAMPLES = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    logging.error(f"Failed to load system agent few-shot examples: {e}. Exiting.")
    exit()

try:
    with open("../prompts/precise_user_template.txt", 'r', encoding='utf-8') as f:
        PRECISE_ANSWERER_PROMPT_TEMPLATE = f.read()
except FileNotFoundError as e:
    logging.error(f"Failed to load precise user prompt template: {e}. Exiting.")
    exit()
    
try:
    with open("../prompts/fuzzy_user_template.txt", 'r', encoding='utf-8') as f:
        FUZZY_ANSWERER_PROMPT_TEMPLATE = f.read()
except FileNotFoundError as e:
    logging.error(f"Failed to load fuzzy user prompt template: {e}. Exiting.")
    exit()


# --- Question Generation with LLM (ReAct Loop) ---
def generate_question_with_llm(pmv_point: float, user_profile: dict, system_agent_instance: Agent, max_attempts: int = 3) -> list[dict]:
    """
    Generates a natural language question and representative scenario using an LLM,
    employing a ReAct (Thought-Action-Observation) loop to ensure the scenario's
    calculated PMV falls within the target range for the user's specific parameters.

    Args:
        pmv_point (float): The target PMV value around which to generate the question.
        user_profile (dict): The specific user's thermal profile (clo, met, etc.).
        system_agent_instance (Agent): An instance of the system LLM agent.
        max_attempts (int): Maximum number of attempts for the ReAct loop.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents one attempt
                    by the system LLM to generate a question and scenario.
                    Each dict includes 'attempt_number', 'attempt_was_successful',
                    'pmv_range', 'feeling', 'raw_response', 'calculated_pmv',
                    'scenario', and 'question_text'.
    """
    pmv_range = map_pmv_point_to_range(pmv_point)
    pmv_min, pmv_max = pmv_range
    feeling_description = get_pmv_feeling_description(pmv_point)

    user_clo = user_profile["D5.Clothing Insulation (clo)"]
    user_met = user_profile["D6.Metabolic Rate (met)"]

    initial_prompt_context = f"""
    Target PMV Range: [{pmv_min:.1f}, {pmv_max:.1f}]
    Thermal Sensation Description: {feeling_description}
    User CLO: {user_clo:.2f}, User MET: {user_met:.2f}

    Please output your Thought and Action.
    The Action part MUST be in strict JSON format, enclosed by triple backticks and marked as 'json'.
    """

    feedback_message = "" # Initialize feedback message for the ReAct loop
    
    # List to store all attempts for this question generation
    all_generation_attempts = [] 
    
    # Variable to hold the successfully generated info, if any
    successful_generated_info = None 

    for attempt_num in range(1, max_attempts + 1):
        logging.info(f"\n🔁 ReAct Attempt {attempt_num}: Generating Question + Scenario")
        
        user_input_to_llm = initial_prompt_context
        if attempt_num > 1:
            user_input_to_llm += f"\nObservation: {feedback_message}\nBased on this observation, please refine your Thought and Action."

        raw_response = system_agent_instance.generate_response(
            user_input=user_input_to_llm,
            few_shot_examples=SYSTEM_AGENT_FEW_SHOT_EXAMPLES
        )
        logging.info(f"🧠 LLM Raw Response: {raw_response}")
        
        parsed_data = extract_json_block(raw_response)
        question_text = extract_question_text(parsed_data) 
        scenario = extract_scenario(parsed_data)

        current_air_temp = scenario.get("air_temp", 24.0)
        current_mrt = scenario.get("mrt", 24.0)
        current_humidity = scenario.get("humidity", ASSUMED_RH_PERCENT)
        current_air_vel = scenario.get("air_vel", ASSUMED_AIR_VELOCITY_MPS)

        calculated_pmv = calculate_pmv(
            air_temp=current_air_temp,
            mrt=current_mrt,
            rh=current_humidity,
            air_vel=current_air_vel,
            clo=user_clo,
            met=user_met
        )

        current_attempt_details = {
            "attempt_number": attempt_num,
            "pmv_range": [float(pmv_min), float(pmv_max)],
            "feeling": feeling_description,
            "raw_response": raw_response,
            "question_text": question_text,
            "scenario": {k: float(v) for k, v in scenario.items()}, # Ensure float for JSON
            "calculated_pmv": float(calculated_pmv) if calculated_pmv is not None else None,
            "attempt_was_successful": False # Default to False, update if successful
        }

        if calculated_pmv is None:
            logging.warning(f"❌ ReAct Attempt {attempt_num}: PMV calculation failed for generated scenario. Retrying.")
            feedback_message = (
                f"The previously generated scenario parameters were invalid or led to a PMV calculation error. "
                f"Please ensure `air_temp`, `mrt`, `humidity`, `air_vel` are reasonable numbers. "
                f"Also, confirm that the `air_temp` and `mrt` are within 10-40°C. "
                f"Remember the user's CLO ({user_clo:.2f}) and MET ({user_met:.2f}) are constant. "
                f"If you included any comments in the JSON, please remove them."
            )
            current_attempt_details["attempt_was_successful"] = False
            all_generation_attempts.append(current_attempt_details)
            continue

        logging.info(f"🧠 LLM Generated Question: {question_text}")
        logging.info(f"📊 Calculated PMV for Scenario (with user's CLO={user_clo:.2f}, MET={user_met:.2f}): {calculated_pmv:.2f}")

        if pmv_min <= calculated_pmv <= pmv_max:
            logging.info("✅ Scenario PMV within target range. Accepting this question.")
            current_attempt_details["attempt_was_successful"] = True
            all_generation_attempts.append(current_attempt_details)
            successful_generated_info = current_attempt_details # Store the successful one
            break # Exit the ReAct loop as a valid question was generated
        else:
            logging.warning(f"❌ ReAct Attempt {attempt_num}: Calculated PMV {calculated_pmv:.2f} is outside target range {pmv_range}. Providing feedback and retrying.")
            
            target_center = (pmv_min + pmv_max) / 2
            pmv_deviation = calculated_pmv - target_center
            suggested_temp_change_heuristic = -pmv_deviation / 0.35

            temp_for_pmv_min_boundary = find_temp_for_pmv(pmv_min, clo=user_clo, met=user_met, rh=current_humidity, air_vel=current_air_vel)
            temp_for_pmv_max_boundary = find_temp_for_pmv(pmv_max, clo=user_clo, met=user_met, rh=current_humidity, air_vel=current_air_vel)
            
            feedback_message = (
                f"Your last generated scenario resulted in a calculated PMV of {calculated_pmv:.2f} "
                f"for the user (CLO: {user_clo:.2f}, MET: {user_met:.2f}). "
                f"The target PMV range is {pmv_range}. "
            )

            temp_reference_str = ""
            if temp_for_pmv_min_boundary is not None and temp_for_pmv_max_boundary is not None:
                ref_min_temp = min(temp_for_pmv_min_boundary, temp_for_pmv_max_boundary)
                ref_max_temp = max(temp_for_pmv_min_boundary, temp_for_pmv_max_boundary)
                temp_reference_str = (
                    f"For this target PMV range, the `air_temp` and `mrt` should ideally be between "
                    f"approximately {ref_min_temp:.1f}°C and {ref_max_temp:.1f}°C. "
                )

            if pmv_deviation > 0:
                feedback_message += (
                    f"This PMV is {abs(pmv_deviation):.2f} units above the target range. "
                    f"Please **lower** the `air_temp` and `mrt` values in your `scenario` by approximately {abs(suggested_temp_change_heuristic):.1f}°C. "
                )
            else:
                feedback_message += (
                    f"This PMV is {abs(pmv_deviation):.2f} units below the target range. "
                    f"Please **raise** the `air_temp` and `mrt` values in your `scenario` by approximately {abs(suggested_temp_change_heuristic):.1f}°C. "
                )
            
            feedback_message += (
                temp_reference_str +
                f"Ensure the new `air_temp` and `mrt` values are within the valid ASHRAE range (10-40°C) "
                f"and that `air_vel` ({current_air_vel} m/s) and `humidity` ({current_humidity}%) are appropriate for the desired sensation."
            )
            logging.warning(f"Feedback to LLM: {feedback_message}")
            current_attempt_details["attempt_was_successful"] = False
            all_generation_attempts.append(current_attempt_details) # Append this failed attempt too

    if not successful_generated_info:
        logging.warning("⚠️ Failed to generate a satisfactory question after multiple attempts. Returning all attempts, the last one will be used.")
    
    return all_generation_attempts # Return the list of all attempts

# --- Bayesian Belief Initialization and Update ---
pmv_values = np.arange(PMV_MIN_VALUE, PMV_MAX_VALUE + PMV_STEP_SIZE, PMV_STEP_SIZE)
initial_prior_probabilities = np.ones_like(pmv_values) / len(pmv_values)

def update_bayesian_belief_soft(prior_probs: np.ndarray, omega: float, confidence: float, target_pmv_range: list[float], tau: float, lambda_threshold: float) -> np.ndarray:
    """
    Updates the Bayesian belief distribution over PMV values based on user feedback.
    Incorporates confidence filtering and positive/negative/neutral feedback modeling.

    Args:
        prior_probs (np.ndarray): The current prior probability distribution over PMV values.
        omega (float): Preference strength (0.0 for contradiction, 1.0 for entailment, 0.5 for neutral).
        confidence (float): Semantic confidence of the NLI model (0.0 to 1.0).
        target_pmv_range (list[float]): The [min, max] PMV range associated with the question asked.
        tau (float): Sensitivity parameter for the Gaussian likelihood (standard deviation).
        lambda_threshold (float): Confidence threshold. If confidence is below this, omega defaults to 0.5.

    Returns:
        np.ndarray: The updated posterior probability distribution.
    """
    if confidence < lambda_threshold:
        logging.info(f"NLI confidence ({confidence:.2f}) below threshold ({lambda_threshold:.2f}). Treating as neutral feedback.")
        omega = 0.5

    mu = (target_pmv_range[0] + target_pmv_range[1]) / 2

    # Revised likelihood calculation for positive and negative feedback
    if omega > 0.55: # Positive feedback: increase probability around mu
        likelihood_gaussian = np.exp(-0.5 * ((pmv_values - mu) / tau) ** 2)
        # Scale strength: omega=1.0 gives full positive effect, omega=0.55 gives minimal positive effect
        strength_factor = (omega - 0.55) / (1.0 - 0.55) # Scales from 0 to 1
        # Mix with a uniform likelihood to ensure prior info isn't completely overridden,
        # and to allow for less strong positive signals.
        likelihood = 1.0 + strength_factor * (likelihood_gaussian - 1.0)
        
    elif omega < 0.45: # Negative feedback: decrease probability around mu
        penalty_gaussian = np.exp(-0.5 * ((pmv_values - mu) / tau) ** 2)
        # Scale penalty strength: omega=0.0 gives full penalty, omega=0.45 gives minimal penalty
        strength_factor = (0.45 - omega) / (0.45 - 0.0) # Scales from 0 to 1
        # Subtract penalty from a uniform likelihood of 1.0
        likelihood = 1.0 - strength_factor * penalty_gaussian
        
    else: # Neutral feedback: likelihood is uniform (no change)
        likelihood = np.ones_like(pmv_values)

    # Ensure likelihood values are never zero or negative to prevent division by zero or numerical issues
    likelihood = np.clip(likelihood, 1e-6, None) 

    posterior = prior_probs * likelihood
    posterior /= posterior.sum() # Normalize to maintain a valid probability distribution

    return posterior

def get_inferred_pmv_range(probabilities: np.ndarray, confidence_level: float = 0.8) -> tuple[float, float]:
    """
    Infers the PMV range from the probability distribution, aiming for a specified
    confidence level. It expands from the peak probability, prioritizing expansion
    towards the weighted mean.

    Args:
        probabilities (np.ndarray): The PMV probability distribution.
        confidence_level (float): The desired confidence level for the PMV range.

    Returns:
        tuple[float, float]: The inferred [min_pmv, max_pmv] range.
    """
    if np.sum(probabilities) == 0:
        return PMV_MIN_VALUE, PMV_MAX_VALUE

    weighted_mean = np.sum(probabilities * pmv_values) / np.sum(probabilities)

    peak_idx = np.argmax(probabilities)
    current_prob_sum = probabilities[peak_idx]
    selected_indices = {peak_idx}

    left_idx = peak_idx - 1
    right_idx = peak_idx + 1

    while current_prob_sum < confidence_level and (left_idx >= 0 or right_idx < len(pmv_values)):
        prob_left = probabilities[left_idx] if left_idx >= 0 else -1
        prob_right = probabilities[right_idx] if right_idx < len(pmv_values) else -1

        # Determine which side to expand based on probability and proximity to weighted mean
        can_expand_left = left_idx >= 0
        can_expand_right = right_idx < len(pmv_values)

        if not can_expand_left and not can_expand_right:
            break # No more indices to expand

        if can_expand_left and can_expand_right:
            # Prefer expanding towards the weighted mean, then by higher probability
            dist_left = abs(pmv_values[left_idx] - weighted_mean)
            dist_right = abs(pmv_values[right_idx] - weighted_mean)

            if prob_left > prob_right:
                if dist_left <= dist_right or prob_left > prob_right * 1.5: # Consider a threshold for significant prob diff
                    expand_to_left = True
                else:
                    expand_to_left = False
            else: # prob_right >= prob_left
                if dist_right < dist_left or prob_right > prob_left * 1.5:
                    expand_to_left = False
                else:
                    expand_to_left = True
        elif can_expand_left:
            expand_to_left = True
        else: # can_expand_right
            expand_to_left = False
        
        if expand_to_left:
            current_prob_sum += probabilities[left_idx]
            selected_indices.add(left_idx)
            left_idx -= 1
        else: # expand_right
            current_prob_sum += probabilities[right_idx]
            selected_indices.add(right_idx)
            right_idx += 1

    if not selected_indices:
        return PMV_MIN_VALUE, PMV_MAX_VALUE

    min_pmv_idx = min(selected_indices)
    max_pmv_idx = max(selected_indices)

    min_pmv = pmv_values[min_pmv_idx]
    max_pmv = pmv_values[max_pmv_idx]

    return min_pmv, max_pmv

def compute_entropy(prob_dist: np.ndarray) -> float:
    """
    Computes the Shannon entropy of a probability distribution.

    Args:
        prob_dist (np.ndarray): The probability distribution.

    Returns:
        float: The entropy value.
    """
    prob_dist = np.clip(prob_dist, 1e-12, 1.0)
    return -np.sum(prob_dist * np.log(prob_dist))

def get_distribution_center(prob_dist: np.ndarray, values: np.ndarray) -> float:
    """
    Calculates the weighted mean (center of mass) of a probability distribution.

    Args:
        prob_dist (np.ndarray): The probability distribution.
        values (np.ndarray): The values corresponding to the probabilities.

    Returns:
        float: The weighted mean.
    """
    if np.sum(prob_dist) == 0:
        return np.mean(values)
    return np.sum(prob_dist * values) / np.sum(prob_dist)

# Function to apply noise to a precise response for 'noisy' user
def apply_noise_to_response(response: str, p_flip: float = NOISE_PROBABILITY) -> str:
    """
    Applies noise to a precise user's response with a given probability.
    For 'noisy' user, a 'Yes' or 'No' response can be flipped to its opposite
    or changed to 'Not sure'. 'Not sure' responses remain unchanged.
    
    Args:
        response (str): The original precise response ("Yes", "No", or "Not sure").
        p_flip (float): The probability of applying noise.

    Returns:
        str: The potentially noisy response.
    """
    if random.random() < p_flip:
        response_lower = response.strip().lower()
        if response_lower in ["yes", "yes.", "yes!"]:
            # With noise probability, 50% chance to flip to "No", 50% chance to become "Not sure"
            return "No" if random.random() < 0.5 else "Not sure"
        elif response_lower in ["no", "no.", "no!"]:
            # With noise probability, 50% chance to flip to "Yes", 50% chance to become "Not sure"
            return "Yes" if random.random() < 0.5 else "Not sure"
        elif response_lower in ["not sure", "not sure."]:
            # 'Not sure' responses are not flipped.
            return "Not sure"
        else:
            # Should not happen with precise user, but handle unexpected input
            logging.warning(f"Unexpected base response for noisy user: '{response}'. No noise applied.")
            return response
    return response

# --- Simulate Dialogue Flow ---
def simulate_dialogue(
    user_profile: dict,
    system_agent_instance: Agent,
    user_agent_instance: Agent,
    user_behavior_type: str,
    max_rounds: int = MAX_DIALOGUE_ROUNDS,
    convergence_threshold_pmv: float = CONVERGENCE_THRESHOLD_PMV,
    output_folder: str = "."
) -> tuple[tuple[float, float], tuple[float | None, float | None], list[dict], int, bool]:
    """
    Simulates a dialogue between a system agent and a user agent to infer user's
    ideal thermal comfort preference (PMV and temperature range).

    Args:
        user_profile (dict): The user's specific profile including actual ideal PMV.
        system_agent_instance (Agent): The LLM agent responsible for asking questions.
        user_agent_instance (Agent): The LLM agent simulating the user's responses.
        user_behavior_type (str): The type of user behavior ('precise', 'fuzzy', 'noisy').
        max_rounds (int): Maximum number of dialogue turns.
        convergence_threshold_pmv (float): PMV range width at which dialogue converges.
        output_folder (str): Path to the folder where plots will be saved.

    Returns:
        tuple: A tuple containing:
            - (final_inferred_min_pmv, final_inferred_max_pmv): The final inferred PMV range.
            - (final_inferred_min_temp, final_inferred_max_temp): The final inferred temperature range.
            - conversation_history (list[dict]): A list of dictionaries detailing each dialogue round.
            - num_rounds (int): The number of rounds actually conducted.
            - pmv_inference_successful (bool): Whether the true PMV was within the inferred range.
    """
    logging.info(f"\n======== Starting Simulation for User ID: {user_profile['ID']} ({user_behavior_type.capitalize()} User) ========")
    logging.info(f"Simulated User's True Ideal PMV (for internal evaluation): {user_profile['Ideal PMV']:.2f}")

    current_pmv_probabilities = initial_prior_probabilities.copy()
    conversation_history = []
    asked_pmv_ranges = set()
    num_rounds = 0

    # User's constant CLO and MET values (for temperature conversion)
    user_clo = user_profile['D5.Clothing Insulation (clo)']
    user_met = user_profile['D6.Metabolic Rate (met)']

    for round_num in range(1, max_rounds + 1):
        num_rounds = round_num
        logging.info(f"\n--- Dialogue Round {round_num} ---")
        best_pmv_point = None

        pmv_center = get_distribution_center(current_pmv_probabilities, pmv_values)
        current_entropy = compute_entropy(current_pmv_probabilities)

        # Strategy for selecting the next PMV point to query
        if round_num <= 1: # Phase 1: Directional Exploration (explore points furthest from current center)
            distances = np.abs(pmv_values - pmv_center)
            for idx in np.argsort(-distances): # Sort by descending distance
                candidate_pmv_point = pmv_values[idx]
                pmv_range = tuple(map_pmv_point_to_range(candidate_pmv_point))
                if pmv_range not in asked_pmv_ranges:
                    best_pmv_point = candidate_pmv_point
                    break
        elif round_num <= 4: # Phase 2: UCB-like Exploration (balance exploiting current knowledge and exploring new ranges)
            beta = 1.5 # Exploration factor
            best_ucb_score = -float('inf')
            for pmv_point in pmv_values:
                pmv_range = tuple(map_pmv_point_to_range(pmv_point))
                if pmv_range in asked_pmv_ranges:
                    continue
                mask = (pmv_values >= pmv_range[0]) & (pmv_values <= pmv_range[1])
                if not np.any(mask):
                    continue # Skip if no PMV values in range

                prob_mass_in_range = current_pmv_probabilities[mask].sum()
                variance_in_range = np.var(current_pmv_probabilities[mask]) if len(current_pmv_probabilities[mask]) > 1 else 0.0
                ucb_score = prob_mass_in_range + beta * np.sqrt(variance_in_range)
                if ucb_score > best_ucb_score:
                    best_ucb_score = ucb_score 
                    best_pmv_point = pmv_point
        else: # Phase 3: Information Gain (focus on reducing uncertainty)
            best_score = -float('inf')
            for pmv_point in pmv_values:
                pmv_range = tuple(map_pmv_point_to_range(pmv_point))
                if pmv_range in asked_pmv_ranges:
                    continue
                mu = np.mean(pmv_range)
                tau = BAYESIAN_TAU
                likelihood = np.exp(-0.5 * ((pmv_values - mu) / tau) ** 2)
                likelihood += 1e-6
                simulated_posterior = current_pmv_probabilities * likelihood
                simulated_posterior /= simulated_posterior.sum()
                posterior_entropy = compute_entropy(simulated_posterior)
                info_gain = current_entropy - posterior_entropy
                
                score = info_gain

                if score > best_score:
                    best_score = score
                    best_pmv_point = pmv_point

        if best_pmv_point is None:
            logging.warning("⚠️ Could not select a valid PMV point for this round. Skipping.")
            break

        # Step 2: System LLM Agent generates question
        all_system_generation_attempts = generate_question_with_llm(
            pmv_point=best_pmv_point,
            user_profile=user_profile,
            system_agent_instance=system_agent_instance,
            max_attempts=3
        )

        # Determine which attempt to use for the current dialogue round
        chosen_attempt_for_round = None
        for attempt_data in all_system_generation_attempts:
            if attempt_data['attempt_was_successful']:
                chosen_attempt_for_round = attempt_data
                break
        if chosen_attempt_for_round is None and all_system_generation_attempts:
            chosen_attempt_for_round = all_system_generation_attempts[-1] # Use the last attempt if no success
        elif not all_system_generation_attempts:
            logging.error("generate_question_with_llm returned an empty list of attempts. Skipping this round.")
            continue

        raw_system_response = chosen_attempt_for_round["raw_response"]
        question_text = chosen_attempt_for_round["question_text"]
        scenario = chosen_attempt_for_round["scenario"]
        target_pmv_range_for_question = chosen_attempt_for_round["pmv_range"]
        selected_pmv_point_for_question = best_pmv_point

        logging.info(f"Selected PMV point for query: {selected_pmv_point_for_question:.2f}")
        logging.info(f"Question's target PMV range: {target_pmv_range_for_question}")
        logging.info(f"System asks: {question_text}")

        asked_pmv_ranges.add(tuple(target_pmv_range_for_question))

        # Step 3: Calculate PMV for the scenario (to simulate user's actual sensation)
        calculated_pmv_for_scenario = calculate_pmv(
            air_temp=scenario["air_temp"],
            mrt=scenario["mrt"],
            rh=scenario["humidity"],
            air_vel=scenario["air_vel"],
            clo=user_clo,
            met=user_met
        )

        if calculated_pmv_for_scenario is None:
            logging.error("PMV calculation failed for the question's scenario. Skipping this round.")
            continue
        
        user_sensation_feeling = get_pmv_feeling_description(calculated_pmv_for_scenario)

        logging.info(f"Calculated PMV for simulated user: {calculated_pmv_for_scenario:.2f}")
        logging.info(f"Simulated user's sensation: {user_sensation_feeling}")

        # Step 4: User LLM Agent generates response based on behavior type
        if user_behavior_type == "fuzzy":
            answerer_prompt = FUZZY_ANSWERER_PROMPT_TEMPLATE
        else: # "precise" and "noisy" users use the precise prompt as base
            answerer_prompt = PRECISE_ANSWERER_PROMPT_TEMPLATE

        full_answerer_prompt = answerer_prompt.format(
            ideal_pmv=user_profile['Ideal PMV'],
            calculated_pmv_for_question=calculated_pmv_for_scenario,
            pmv_feeling_description=user_sensation_feeling,
            question_text=question_text
        )
        
        user_response_from_llm = user_agent_instance.generate_response(full_answerer_prompt)
        logging.info(f"LLM User Agent Raw Response (before re-validation): '{user_response_from_llm}'")

        # --- User Agent Response Re-validation based on numerical rules ---
        abs_pmv_diff = np.abs(calculated_pmv_for_scenario - user_profile['Ideal PMV'])
        re_validated_response = user_response_from_llm 

        # Determine the *intended* strict sentiment based on numerical rules
        intended_strict_sentiment = None
        if abs_pmv_diff <= 0.5:
            intended_strict_sentiment = "Yes"
        elif 0.5 < abs_pmv_diff < 1.0:
            intended_strict_sentiment = "Not sure"
        elif abs_pmv_diff >= 1.0:
            intended_strict_sentiment = "No"

        # Apply re-validation
        if intended_strict_sentiment is not None:
            if user_behavior_type in ["precise", "noisy"]:
                current_response_lower_stripped = re_validated_response.strip().lower()
                if intended_strict_sentiment == "Yes" and current_response_lower_stripped not in ["yes", "yes.", "yes!"]:
                    logging.warning(f"Re-validating precise/noisy user: PMV diff {abs_pmv_diff:.2f} <= 0.5. Forcing to 'Yes'. Original: '{re_validated_response}'")
                    re_validated_response = "Yes"
                elif intended_strict_sentiment == "Not sure" and current_response_lower_stripped not in ["not sure", "not sure."]:
                    logging.warning(f"Re-validating precise/noisy user: PMV diff {abs_pmv_diff:.2f} (0.5-1.0). Forcing to 'Not sure'. Original: '{re_validated_response}'")
                    re_validated_response = "Not sure"
                elif intended_strict_sentiment == "No" and current_response_lower_stripped not in ["no", "no.", "no!"]:
                    logging.warning(f"Re-validating precise/noisy user: PMV diff {abs_pmv_diff:.2f} >= 1.0. Forcing to 'No'. Original: '{re_validated_response}'")
                    re_validated_response = "No"
            
            elif user_behavior_type == "fuzzy":
                contains_positive_kw = contains_word_match(re_validated_response, FUZZY_POSITIVE_KEYWORDS + ["yes", "yes.", "yes!"])
                contains_neutral_kw = contains_word_match(re_validated_response, FUZZY_NEUTRAL_KEYWORDS + ["not sure", "not sure."])
                contains_negative_kw = contains_word_match(re_validated_response, FUZZY_NEGATIVE_KEYWORDS + ["no", "no.", "no!"])

                is_contradictory = (contains_positive_kw and contains_negative_kw) or \
                                   (contains_positive_kw and contains_neutral_kw) or \
                                   (contains_negative_kw and contains_neutral_kw)
                
                force_correction = False
                if is_contradictory:
                    force_correction = True
                    logging.warning(f"Re-validating fuzzy user: Response is contradictory ('{re_validated_response}'). Forcing intent.")
                elif intended_strict_sentiment == "Yes" and not contains_positive_kw:
                    force_correction = True
                elif intended_strict_sentiment == "Not sure" and not contains_neutral_kw:
                    force_correction = True
                elif intended_strict_sentiment == "No" and not contains_negative_kw:
                    force_correction = True

                if force_correction:
                    if intended_strict_sentiment == "Yes":
                        logging.warning(f"Re-validating fuzzy user: PMV diff {abs_pmv_diff:.2f} <= 0.5. Forcing positive intent. Original: '{re_validated_response}'")
                        re_validated_response = random.choice(FUZZY_POSITIVE_PHRASES)
                    elif intended_strict_sentiment == "Not sure":
                        logging.warning(f"Re-validating fuzzy user: PMV diff {abs_pmv_diff:.2f} (0.5-1.0). Forcing neutral intent. Original: '{re_validated_response}'")
                        re_validated_response = random.choice(FUZZY_NEUTRAL_PHRASES)
                    elif intended_strict_sentiment == "No":
                        logging.warning(f"Re-validating fuzzy user: PMV diff {abs_pmv_diff:.2f} >= 1.0. Forcing negative intent. Original: '{re_validated_response}'")
                        re_validated_response = random.choice(FUZZY_NEGATIVE_PHRASES)
        
        user_response = re_validated_response
        logging.info(f"LLM User Agent Response (after re-validation): '{user_response}'")
        
        # --- End User Agent Response Re-validation ---

        # Step 4b: Apply noise if user_behavior_type is 'noisy'
        original_user_response = user_response 
        if user_behavior_type == "noisy":
            user_response = apply_noise_to_response(user_response, NOISE_PROBABILITY)
            logging.info(f"LLM User Agent Noisy Response (applied noise): '{user_response}' (Original: '{original_user_response}')")

        # Step 5: NLI analysis of user's response
        omega, confidence = nli_soft_label(question_text, user_response, model_type='deberta_enhanced', user_behavior_type=user_behavior_type)
        logging.info(f"NLI Analysis Result: omega={omega:.2f}, confidence={confidence:.2f}")

        # Step 6: Bayesian belief update
        current_pmv_probabilities = update_bayesian_belief_soft(
            prior_probs=current_pmv_probabilities,
            omega=omega,
            confidence=confidence,
            target_pmv_range=target_pmv_range_for_question,
            tau=BAYESIAN_TAU,
            lambda_threshold=BAYESIAN_LAMBDA_THRESHOLD
        )
        
        # Dynamic confidence_level for get_inferred_pmv_range
        if round_num <= 5: # Early rounds, broader range to capture more possibilities
            current_inference_confidence_for_range = INFERENCE_CONFIDENCE_LEVEL
        elif round_num == MAX_DIALOGUE_ROUNDS: # Last round, aim for a tighter inferred range
            current_inference_confidence_for_range = INFERENCE_CONFIDENCE_LEVEL_DYNAMIC

        inferred_min_pmv, inferred_max_pmv = get_inferred_pmv_range(current_pmv_probabilities, confidence_level=current_inference_confidence_for_range)
        current_pmv_range_width = inferred_max_pmv - inferred_min_pmv
        logging.info(f"Current inferred PMV range (using confidence {current_inference_confidence_for_range:.1f}): [{inferred_min_pmv:.2f}, {inferred_max_pmv:.2f}] (Width: {current_pmv_range_width:.2f})")
        
        inferred_min_temp, inferred_max_temp = convert_pmv_range_to_temp_range(
            inferred_min_pmv, inferred_max_pmv,
            user_clo,
            user_met
        )
        
        if inferred_min_temp is not None and inferred_max_temp is not None:
             logging.info(f"Current inferred temperature range: [{inferred_min_temp:.2f}℃, {inferred_max_temp:.2f}℃]")
        else:
            logging.warning("⚠️ Could not infer current temperature range (PMV inverse calculation failed).")


        # --- Store all valuable information for this round ---
        current_temp_values_for_dist = []
        current_temp_probs_for_dist = []
        for i, pmv in enumerate(pmv_values):
            temp_at_pmv, _ = convert_pmv_range_to_temp_range(pmv, pmv, user_clo, user_met)
            if temp_at_pmv is not None:
                current_temp_values_for_dist.append(float(temp_at_pmv))
                current_temp_probs_for_dist.append(float(current_pmv_probabilities[i]))

        conversation_entry = {
            "round": round_num,
            "system_llm_generation_attempts": all_system_generation_attempts,
            "system_ask_chosen_for_this_round": {
                "question_text": question_text,
                "target_pmv_point_for_question": float(selected_pmv_point_for_question),
                "target_pmv_range_for_question": [float(target_pmv_range_for_question[0]), float(target_pmv_range_for_question[1])],
                "scenario_details": {k: float(v) for k, v in scenario.items()},
                "calculated_pmv_for_scenario": float(calculated_pmv_for_scenario),
                "user_sensation_feeling_for_scenario": user_sensation_feeling,
                "raw_system_llm_response_of_chosen_attempt": raw_system_response
            },
            "user_response_details": {
                "user_response": user_response,
                "original_user_response": original_user_response if user_behavior_type == "noisy" else None
            },
            "nli_analysis_result": {
                "omega": float(omega),
                "confidence": float(confidence)
            },
            "bayesian_update_state": {
                "current_pmv_distribution_snapshot": current_pmv_probabilities.copy().tolist(),
                "current_temp_distribution_snapshot_values": current_temp_values_for_dist,
                "current_temp_distribution_snapshot_probs": current_temp_probs_for_dist,
                "inferred_pmv_range": [float(inferred_min_pmv), float(inferred_max_pmv)],
                "inferred_pmv_range_width": float(current_pmv_range_width),
                "inferred_temp_range": [float(inferred_min_temp) if inferred_min_temp is not None else None, float(inferred_max_temp) if inferred_max_temp is not None else None]
            }
        }
        conversation_history.append(conversation_entry)

        # Convergence check
        if current_pmv_range_width <= convergence_threshold_pmv:
            logging.info(f"Convergence threshold ({convergence_threshold_pmv:.2f}) reached. Dialogue ending.")
            break
            
    logging.info("\n--- Dialogue Ended ---")

    final_inferred_min_pmv, final_inferred_max_pmv = get_inferred_pmv_range(current_pmv_probabilities, confidence_level=current_inference_confidence_for_range)
    logging.info(f"System's final inferred user ideal PMV preference range: [{final_inferred_min_pmv:.2f}, {final_inferred_max_pmv:.2f}]")
    
    final_inferred_min_temp, final_inferred_max_temp = convert_pmv_range_to_temp_range(
        final_inferred_min_pmv, final_inferred_max_pmv,
        user_clo,
        user_met
    )
    
    if final_inferred_min_temp is not None and final_inferred_max_temp is not None:
        logging.info(f"System's final inferred user ideal indoor temperature preference range: [{final_inferred_min_temp:.2f}℃, {final_inferred_max_temp:.2f}℃]")
    else:
        logging.warning("⚠️ Could not infer user's ideal indoor temperature preference range (PMV inverse calculation failed).")

    # Evaluate accuracy
    true_ideal_temp = find_temp_for_pmv(user_profile['Ideal PMV'], 
                                        user_clo,
                                        user_met)

    pmv_success = (final_inferred_min_pmv <= user_profile['Ideal PMV'] <= final_inferred_max_pmv) or \
                  (np.isclose(user_profile['Ideal PMV'], final_inferred_min_pmv, atol=PMV_STEP_SIZE/2)) or \
                  (np.isclose(user_profile['Ideal PMV'], final_inferred_max_pmv, atol=PMV_STEP_SIZE/2))

    if pmv_success:
        logging.info("PMV inference successful: True PMV is within the inferred PMV range.")
        logging.info(f"True Ideal PMV: {user_profile['Ideal PMV']:.2f}")
    else:
        logging.info("PMV inference failed: True PMV is NOT within the inferred PMV range.")
        logging.info(f"True Ideal PMV: {user_profile['Ideal PMV']:.2f}")

    if final_inferred_min_temp is not None and final_inferred_max_temp is not None and true_ideal_temp is not None:
        temp_success = (final_inferred_min_temp <= true_ideal_temp <= final_inferred_max_temp) or \
                       (np.isclose(true_ideal_temp, final_inferred_min_temp, atol=0.1)) or \
                       (np.isclose(true_ideal_temp, final_inferred_max_temp, atol=0.1))
        if temp_success:
            logging.info("Temperature inference successful: True ideal temperature is within the inferred temperature range.")
            logging.info(f"True Ideal Temperature: {true_ideal_temp:.2f}℃")
        else:
            logging.info("Temperature inference failed: True ideal temperature is NOT within the inferred temperature range.")
            logging.info(f"True Ideal Temperature: {true_ideal_temp:.2f}℃")
    else:
        logging.info("Could not determine valid temperature range or true ideal temperature for evaluation.")

    logging.info(f"======== Simulation for User ID: {user_profile['ID']} ({user_behavior_type.capitalize()} User) Ended ========\n")

    # Plotting PMV probability distribution
    plt.figure(figsize=(12, 6))
    plt.plot(pmv_values, current_pmv_probabilities, 'b-', linewidth=2)
    plt.fill_between(pmv_values, 0, current_pmv_probabilities, alpha=0.2)
    plt.axvline(user_profile['Ideal PMV'], color='r', linestyle='--', label=f'True Ideal PMV: {user_profile["Ideal PMV"]:.2f}')
    plt.title(f'Final PMV Distribution for User {user_profile["ID"]} ({user_behavior_type.capitalize()})')
    plt.xlabel('PMV')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'user_{user_profile["ID"]}_{user_behavior_type}_pmv_distribution.png')) 
    plt.close()

    # Plotting Temperature distribution
    valid_temp_values = []
    valid_probs = []
    for i, pmv in enumerate(pmv_values):
        temp_at_pmv, _ = convert_pmv_range_to_temp_range(pmv, pmv, 
                                                            user_clo,
                                                            user_met)
        if temp_at_pmv is not None:
            valid_temp_values.append(temp_at_pmv)
            valid_probs.append(current_pmv_probabilities[i])
    
    if valid_temp_values:
        plt.figure(figsize=(12, 6))
        sorted_indices = np.argsort(valid_temp_values)
        plt.plot(np.array(valid_temp_values)[sorted_indices], np.array(valid_probs)[sorted_indices], 'r-', linewidth=2)
        if true_ideal_temp is not None:
             plt.axvline(true_ideal_temp, color='b', linestyle='--', label=f'True Ideal Temp: {true_ideal_temp:.2f}℃')
        plt.title(f'Temperature Distribution for User {user_profile["ID"]} ({user_behavior_type.capitalize()})')
        plt.xlabel('Temperature (℃)')
        plt.ylabel('Probability Density')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'user_{user_profile["ID"]}_{user_behavior_type}_temp_distribution.png'))
        plt.close()
    else:
        logging.warning(f"Could not plot temperature distribution for User {user_profile['ID']} ({user_behavior_type.capitalize()}) due to conversion issues.")

    # Convert numpy types to native Python floats for clean output
    final_pmv_range_out = (float(final_inferred_min_pmv), float(final_inferred_max_pmv))
    final_temp_range_out = (
        float(final_inferred_min_temp) if final_inferred_min_temp is not None else None, 
        float(final_inferred_max_temp) if final_inferred_max_temp is not None else None
    )
    pmv_inference_successful = bool(pmv_success)

    return final_pmv_range_out, final_temp_range_out, conversation_history, num_rounds, pmv_inference_successful

# --- Main Execution ---
if __name__ == "__main__":
    logging.info(f"Loading preprocessed data from: {PROCESSED_DATA_INPUT_PATH}")
    try:
        ctcd_df = pd.read_csv(PROCESSED_DATA_INPUT_PATH, encoding='utf-8-sig')
        logging.info(f"Successfully loaded {len(ctcd_df)} preprocessed records.")
        logging.info("First 5 rows of the loaded preprocessed data:")
        logging.info(ctcd_df.head())
    except FileNotFoundError:
        logging.error(f"Error: Preprocessed data file not found at {PROCESSED_DATA_INPUT_PATH}. Please ensure the data file exists in the 'data/' folder.")
        exit()
    except Exception as e:
        logging.error(f"Error loading preprocessed data: {e}")
        exit()

    if not ctcd_df.empty:
        os.makedirs(OUTPUT_FOLDER_NAME, exist_ok=True)
        logging.info(f"Saving all results to folder: {OUTPUT_FOLDER_NAME}")

        # Initialize System Agent once
        system_agent_instance = Agent(SYSTEM_AGENT_CONFIG_PATH)

        # Initialize all types of User Agents once
        user_agents = {}
        for behavior_type, config_path in USER_AGENT_CONFIG_PATHS.items():
            user_agents[behavior_type] = Agent(config_path)

        valid_users = ctcd_df['ID'].unique().tolist()
        user_random_seed = 24
        rng = random.Random(user_random_seed)

        num_users_for_experiment = 100 # Set the number of users for the experiment
        
        if len(valid_users) < num_users_for_experiment:
            logging.warning(f"Only {len(valid_users)} unique users available, but {num_users_for_experiment} requested. Using all available users.")
            selected_user_ids = valid_users
        else:
            selected_user_ids = rng.sample(valid_users, num_users_for_experiment)

        for user_id in selected_user_ids:
            simulated_user_profile = ctcd_df[ctcd_df['ID'] == user_id].iloc[0].to_dict()
            true_ideal_temp_for_user = find_temp_for_pmv(simulated_user_profile['Ideal PMV'],
                                                          simulated_user_profile['D5.Clothing Insulation (clo)'],
                                                          simulated_user_profile['D6.Metabolic Rate (met)'])

            for behavior_type in USER_BEHAVIOR_TYPES:
                logging.info(f"\n===== Running Simulation for User: {user_id}, Type: {behavior_type.capitalize()} =====")
                
                current_user_agent_instance = user_agents[behavior_type]

                final_pmv_range, final_temp_range, conv_history, num_rounds_taken, pmv_success_flag = simulate_dialogue(
                    simulated_user_profile,
                    system_agent_instance,
                    current_user_agent_instance,
                    user_behavior_type=behavior_type,
                    max_rounds=MAX_DIALOGUE_ROUNDS,
                    convergence_threshold_pmv=CONVERGENCE_THRESHOLD_PMV,
                    output_folder=OUTPUT_FOLDER_NAME
                )
                
                # Immediately save the results for the current user and behavior type to a separate JSON file
                current_user_full_results = {
                    "user_id": user_id,
                    "user_behavior_type": behavior_type,
                    "true_ideal_pmv": simulated_user_profile['Ideal PMV'],
                    "inferred_min_pmv": final_pmv_range[0],
                    "inferred_max_pmv": final_pmv_range[1],
                    "true_ideal_temp": true_ideal_temp_for_user,
                    "inferred_min_temp": final_temp_range[0],
                    "inferred_max_temp": final_temp_range[1],
                    "pmv_inference_successful": pmv_success_flag,
                    "num_rounds_taken": num_rounds_taken,
                    "conversation_history": conv_history
                }
                
                # Generate a unique filename to avoid overwriting
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                # Name the file using user ID and behavior type for easy lookup
                single_result_filename = f"results_user_{user_id}_{behavior_type}_{timestamp}.json"
                single_result_path = os.path.join(OUTPUT_FOLDER_NAME, single_result_filename)
                
                try:
                    with open(single_result_path, 'w', encoding='utf-8') as f:
                        json.dump(current_user_full_results, f, indent=4, ensure_ascii=False)
                    logging.info(f"Results for User {user_id} ({behavior_type}) saved to: {single_result_path}")
                except Exception as e:
                    logging.error(f"Failed to save results for User {user_id} ({behavior_type}) to {single_result_path}: {e}")

                # After each simulation, try to clear memory, especially when using GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect() # Explicitly call garbage collection

        logging.info(f"\nAll individual simulation results saved to folder: {OUTPUT_FOLDER_NAME}")

        # --- Logic to rebuild the summary results ---
        all_final_summaries = []
        for filename in os.listdir(OUTPUT_FOLDER_NAME):
            # Ensure we only load the simulation results files we just saved
            if filename.startswith("results_user_") and filename.endswith(".json"):
                file_path = os.path.join(OUTPUT_FOLDER_NAME, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Extract only the information needed for the summary, skipping conversation_history to avoid loading large data into memory again
                        summary_data = {k: data[k] for k in data if k != "conversation_history"}
                        all_final_summaries.append(summary_data)
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON from {filename}: {e}")
                except Exception as e:
                    logging.error(f"Error processing file {filename}: {e}")

        if all_final_summaries:
            results_df = pd.DataFrame(all_final_summaries)
            logging.info("\n--- Overall Simulation Results Summary ---")
            logging.info(results_df.groupby('user_behavior_type').agg(
                avg_pmv_success=('pmv_inference_successful', lambda x: x.mean() * 100),
                avg_rounds=('num_rounds_taken', 'mean'),
                avg_inferred_pmv_range_width=('inferred_max_pmv', lambda x: (x - results_df.loc[x.index, 'inferred_min_pmv']).mean()),
                median_inferred_pmv_range_width=('inferred_max_pmv', lambda x: (x - results_df.loc[x.index, 'inferred_min_pmv']).median())
            ).round(2))

            results_df['inferred_pmv_center'] = (results_df['inferred_min_pmv'] + results_df['inferred_max_pmv']) / 2
            results_df['pmv_mae'] = (results_df['inferred_pmv_center'] - results_df['true_ideal_pmv']).abs()
            
            logging.info("\n--- Mean Absolute Error (PMV) by User Behavior Type ---")
            logging.info(results_df.groupby('user_behavior_type')['pmv_mae'].mean().round(3))

            logging.info("\n--- Head of All Simulation Results ---")
            logging.info(results_df.head())
        else:
            logging.warning("No simulation results found to summarize. Check if JSON files were saved correctly.")

    else:
        logging.info("Preprocessed data loaded but is empty. Cannot run simulation.")