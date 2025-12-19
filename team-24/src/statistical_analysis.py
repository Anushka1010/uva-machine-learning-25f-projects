import pandas as pd
import matplotlib.pyplot as plt

from bert import LABELS

def main() -> None:
    df = pd.read_csv("bert_predictions.csv")

    # print coefficients
    print("Coefficients:")
    print(f"Positive coefficient: {df['prob_positive'].corr(df['price_change_pct'])}")
    print(f"Neutral coefficient: {df['prob_neutral'].corr(df['price_change_pct'])}")
    print(f"Negative coefficient: {df['prob_negative'].corr(df['price_change_pct'])}")

    # print means and deviations of stock price changes for each sentiment
    print("\nMeans and deviations:")
    for label in LABELS:
        subset = df[df["predicted_label"] == label]["price_change_pct"]
        print(f"{label}: mean={subset.mean()}, std={subset.std()}")

    # plot changes in price for each sentiment with a boxplot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, label in zip(axes, LABELS):
        subset = df[df["predicted_label"] == label]["price_change_pct"]
        ax.boxplot(subset)
        ax.set_title(label)
        ax.set_ylabel("price change (%)")

    fig.suptitle("Boxplot of price change percentage for each sentiment")
    plt.ylim(ymin=-25, ymax=25)
    plt.tight_layout()
    plt.show() 

    # plot changes in price for each sentiment with a scatter plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, label, prob_col in zip(axes, LABELS, [f"prob_{label}" for label in LABELS]):
        subset = df[df["predicted_label"] == label]
        ax.scatter(subset[prob_col], subset["price_change_pct"], alpha=0.6)
        ax.set_title(label)
        ax.set_xlabel(f"{label} probability")
        ax.set_ylabel("price change (%)")

    fig.suptitle("Scatter plot of price change percentage for each sentiment")
    plt.ylim(ymin=-25, ymax=25)
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()