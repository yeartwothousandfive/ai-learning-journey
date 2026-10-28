
import json
import os

MODEL_FILE = "model.json"

training_data = [
    (1, 28),
    (2, 56),
    (4, 112),
    (5, 140)
]

if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "r") as file:
        model = json.load(file)

    weight = model["weight"]
    bias = model["bias"]

    print("Saved AI model loaded!")

else:
    weight = 0.0
    bias = 0.0

    print("Starting a new AI model!")

print("Initial weight:", round(weight, 4))
print("Initial bias:", round(bias, 4))

print("\nChoose an action:")
print("1. Train an AI")
print("2. Test the AI")
print("3. Show learned parameters")

choice = input("Enter 1, 2, or 3: ")

if choice == "1":
    learning_rate = 0.01
    epochs = 2000

    for epoch in range(epochs):
        weight_gradient = 0.0
        bias_gradient = 0.0

        for x, correct_answer in training_data:
            prediction = weight * x + bias
            error = prediction - correct_answer

            weight_gradient += (
                2 * error * x / len(training_data)
            )

            bias_gradient += (
                2 * error / len(training_data)
            )

        weight -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient

    model = {
        "weight": weight,
        "bias": bias
    }

    with open(MODEL_FILE, "w") as file:
        json.dump(model, file, indent=4)

    print("\nTraining finished!")
    print("Weight:", round(weight, 4))
    print("Bias:", round(bias, 4))
    print("Model saved to", MODEL_FILE)

elif choice == "2":
    number = float(input("Enter number: "))

    prediction = weight * number + bias

    print("AI prediction:", round(prediction, 2))

elif choice == "3":
    print("\nLearned Parameters:")
    print("Weight:", weight)
    print("Bias:", bias)

else:
    print("Invalid choice. Please enter 1, 2, or 3.")
