# Tasks 1 and 2: PEAS specifications and environment classifications

peas = {
    "Delivery robot": {
        "Performance measure": [
            "Deliver the correct parcel to the correct destination",
            "Meet delivery-time targets",
            "Avoid collisions and parcel damage",
            "Minimize energy use",
        ],
        "Environment": [
            "Sidewalks and building corridors",
            "Pedestrians and other robots",
            "Curbs, doors, ramps, and elevators",
            "Parcel pickup and drop-off points",
            "Changing weather and lighting",
        ],
        "Actuators": [
            "Drive motors and steering",
            "Brakes",
            "Robotic arm or parcel latch",
            "Status display, speaker, or warning light",
        ],
        "Sensors": [
            "RGB or depth cameras",
            "Lidar or ultrasonic range sensors",
            "Wheel encoders and inertial sensors",
            "GPS outdoors",
            "Parcel-latch and battery sensors",
        ],
        "Constraint or trade-off": (
            "Moving faster can improve delivery time but increases "
            "stopping distance and collision risk."
        ),
    },

    "LLM-based student-support agent": {
        "Performance measure": [
            "Give accurate answers grounded in approved course information",
            "Respond in a useful amount of time",
            "Protect student privacy",
            "Escalate uncertain or sensitive cases",
        ],
        "Environment": [
            "Student chat interface",
            "Course and university support questions",
            "Approved FAQ and policy documents",
            "Support staff and ticket queue",
            "Different student language and accessibility needs",
        ],
        "Actuators": [
            "Send a chat reply",
            "Ask a clarifying question",
            "Create an escalation ticket",
            "Provide an approved document link",
        ],
        "Sensors and inputs": [
            "Student messages and conversation context",
            "Read-only search of approved FAQs",
            "Read-only course catalog and policy search",
            "Status of an existing support ticket",
        ],
        "Operational assumptions": [
            "The agent can read the current conversation.",
            "It can search approved FAQ, course catalog, and policy pages.",
            "It can inspect existing ticket status and create an escalation ticket.",
            "It cannot access grades, private student records, or email accounts.",
            "It cannot change official records or make enrollment decisions.",
        ],
        "Constraint or trade-off": (
            "Searching trusted sources and escalating uncertain answers "
            "can improve accuracy but may increase response time."
        ),
    },
}

for agent, specification in peas.items():
    print(f"\n{agent}")
    for component, entries in specification.items():
        print(f"\n{component}:")
        if isinstance(entries, list):
            for entry in entries:
                print(f"  - {entry}")
        else:
            print(f"  {entries}")


environment_classifications = [
    {
        "system": "Delivery robot — real world",
        "observability": "Partially observable",
        "dynamics": "Stochastic",
        "observable_example": "Camera and lidar readings",
        "hidden_example": "A pedestrian's intent or an occluded obstacle",
        "outcome_example": (
            "A drive command may have different outcomes due to traction, "
            "moving people, or sensor noise."
        ),
    },
    {
        "system": "Student-support agent — real service",
        "observability": "Partially observable",
        "dynamics": "Stochastic",
        "observable_example": "Student message and search results",
        "hidden_example": (
            "Unstated student circumstances or a policy change not yet indexed"
        ),
        "outcome_example": (
            "A reply may resolve the question, prompt clarification, "
            "or require escalation."
        ),
    },
    {
        "system": "Delivery robot — simplified simulator",
        "observability": "Fully observable",
        "dynamics": "Deterministic",
        "observable_example": (
            "Assumption: simulator exposes all modeled positions, obstacles, "
            "battery, and parcel state."
        ),
        "hidden_example": "None in this simplified simulator",
        "outcome_example": (
            "Assumption: fixed movement rules map each command to one result."
        ),
    },
    {
        "system": "Student-support agent — simplified simulator",
        "observability": "Fully observable",
        "dynamics": "Deterministic",
        "observable_example": (
            "Assumption: simulator provides the complete student/query state "
            "and a fixed knowledge base."
        ),
        "hidden_example": "None in this simplified simulator",
        "outcome_example": (
            "Assumption: each agent action maps to a fixed outcome."
        ),
    },
]

print("\nEnvironment classifications")
for item in environment_classifications:
    print(f"\n{item['system']}")
    print(f"  Observability: {item['observability']}")
    print(f"  Dynamics: {item['dynamics']}")
    print(f"  Observable example: {item['observable_example']}")
    print(f"  Hidden example: {item['hidden_example']}")
    print(f"  Action outcome: {item['outcome_example']}")


# Task 3: Room controller with previous-mode memory
#
# Assumptions:
# - Heat below 18°C.
# - Cool above 26°C.
# - At exactly 18°C or 26°C, target IDLE.
# - An empty room targets OFF regardless of temperature.

class RoomController:
    def __init__(self):
        self.previous_target_mode = None

    def update(self, temperature_c, occupied):
        """Return the target mode and whether a command should be sent."""
        previous_mode = self.previous_target_mode

        # Occupancy has priority over temperature.
        if not occupied:
            target_mode = "OFF"
        elif temperature_c < 18:
            target_mode = "HEAT"
        elif temperature_c > 26:
            target_mode = "COOL"
        else:
            target_mode = "IDLE"

        send_command = target_mode != previous_mode
        self.previous_target_mode = target_mode

        return previous_mode, target_mode, send_command


# Cases run in sequence so the controller remembers the previous target.
cases = [
    ("Initial heat", 17, True),
    ("Repeated heat percept", 17, True),
    ("Lower exact boundary", 18, True),
    ("Upper exact boundary", 26, True),
    ("Mode transition to cool", 27, True),
    ("Repeated cool percept", 27, True),
    ("Empty room priority", 30, False),
]

controller = RoomController()

print("\nRoom-controller cases")
for name, temperature, occupied in cases:
    previous, target, send = controller.update(temperature, occupied)

    command_result = "SEND" if send else "NO COMMAND"
    print(
        f"{name}: "
        f"previous_mode={previous}; "
        f"percept={{'temperature_c': {temperature}, "
        f"'occupied': {occupied}}}; "
        f"target={target}; "
        f"outcome={command_result}"
    )