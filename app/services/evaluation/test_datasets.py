REFERENCE_DOC2 = "FK2DW.pdf"

DATASET2 = [
    {
        "question": "What wireless connection methods are available for the mouse?",
        "expected_answer": "The mouse supports wireless connection via a 4K enhanced receiver and a USB receiver.",
        "ground_truth": ["4k enhanced receiver", "usb receiver"],
        "reference_doc": REFERENCE_DOC2,
    },
    {
        "question": "How can the mouse be connected in wired mode?",
        "expected_answer": "The mouse can be connected using a USB cable directly to the computer.",
        "ground_truth": ["usb cable", "wired connection"],
        "reference_doc": REFERENCE_DOC2,
    },
    {
        "question": "What does a flashing amber LED indicate on the receiver?",
        "expected_answer": "A flashing amber LED indicates that the signal is not stable or the mouse is about to disconnect.",
        "ground_truth": ["flashing amber", "unstable signal", "disconnect"],
        "reference_doc": REFERENCE_DOC2,
    },
    {
        "question": "How long does it take to fully charge the mouse?",
        "expected_answer": "It takes about 2 hours to fully charge the mouse.",
        "ground_truth": ["2 hours", "charging time"],
        "reference_doc": REFERENCE_DOC2,
    },
    {
        "question": "What happens when the mouse is connected via USB cable regardless of mode?",
        "expected_answer": "The mouse switches to wired operation and starts charging.",
        "ground_truth": ["wired operation", "charging"],
        "reference_doc": REFERENCE_DOC2,
    },
    {
        "question": "What DPI options are available on the mouse?",
        "expected_answer": "The mouse supports DPI settings of 400, 800, 1000, 1200, 1600, and 3200.",
        "ground_truth": ["400", "800", "1000", "1200", "1600", "3200"],
        "reference_doc": REFERENCE_DOC2,
    },
    {
        "question": "How can you change the polling rate?",
        "expected_answer": "You can change the polling rate by pressing the polling rate button repeatedly to cycle through options.",
        "ground_truth": ["polling rate button", "cycle"],
        "reference_doc": REFERENCE_DOC2,
    },
    {
        "question": "What is the effect of higher polling rates?",
        "expected_answer": "Higher polling rates increase power consumption.",
        "ground_truth": ["higher polling rate", "power consumption"],
        "reference_doc": REFERENCE_DOC2,
    },
    {
        "question": "What is performance mode and its trade-off?",
        "expected_answer": "Performance mode improves tracking responsiveness but increases power consumption.",
        "ground_truth": ["performance mode", "responsive", "power consumption"],
        "reference_doc": REFERENCE_DOC2,
    },
    {
        "question": "What should you do before storing the mouse for a long time?",
        "expected_answer": "You should switch off the mouse and properly pack it after removing the cable.",
        "ground_truth": ["power off", "remove cable", "storage"],
        "reference_doc": REFERENCE_DOC2,
    },
]
