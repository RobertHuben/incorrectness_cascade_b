import numpy as np
from score_data import count_categories_1a, count_categories_1b, load_prompt_abbreviations
import matplotlib.pyplot as plt
import scipy.stats as stats
import pymannkendall as mk
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd


def load_y_1a():
    # loads the y data
    data = np.array(count_categories_1a())
    YY = data[2]/(data[1]+data[2])
    return YY


def load_y_1b():
    # loads the y data
    data = np.array(count_categories_1b())
    YY = data[2]/(data[1]+data[2])
    return YY


def load_n_1a():
    # loads the number of questions answered correctly/incorrectly (as opposed to misformatted)
    data = np.array(count_categories_1a())
    NN = data[1]+data[2]
    return NN


def load_n_1b():
    # loads the number of questions answered correctly/incorrectly (as opposed to misformatted)
    data = np.array(count_categories_1b())
    NN = data[1]+data[2]
    return NN


def preliminary_test_most_misformatted():
    # says the highest concentration of misformatted answers
    data = np.array(count_categories_1b())
    percent_misformatted = data[3]/data[0]
    print(np.max(percent_misformatted))


def test_1b():
    # "Perform a multiple-regression analysis of Y on X and (dummy-coded values of) P, with interaction terms between X and P. For this analysis, we will only consider P taking the values g-j. In particular, we will look for statistically significant interaction terms between X and P."
    y_with_q_0 = load_y_1a()
    y_with_q_1 = load_y_1b()

    for P in range(10):

        # Create a DataFrame for the independent variables (X, Q, and interaction terms)
        X = np.arange(11)

        # Create a dataset for the analysis
        data = pd.DataFrame({'Y': np.concatenate(
            (y_with_q_0[P], (y_with_q_1[P]))), 'X': np.tile(X, 2), 'Q': np.repeat([0, 1], 11)})

        # Create interaction terms
        data['X_Q'] = data['X'] * data['Q']

        # Perform the multiple regression analysis
        model = smf.ols(
            'Y ~ X + Q + X_Q', data=data).fit()

        # Print the results

        print_statement = f"P={P+1}," + ','.join(
            [f"{model.params[param]:.4f}({model.pvalues[param]:.4f})" for param in list(model.params.index)])
        print(print_statement)


def make_comparisons_graph():
    # Load data
    data_1a = load_y_1a()
    data_1b = load_y_1b()
    prompt_abbreviations = load_prompt_abbreviations()

    # Set up the figure and the 3x4 grid of subplots
    fig, axes = plt.subplots(nrows=3, ncols=4)

    # Loop through the 10 prompts and create the line graphs in the subplots
    for i in range(10):
        # Calculate the row and column index for the current subplot
        row = i // 4
        col = i % 4

        # Extract the i-th row from A and B
        row_A = data_1a[i]
        row_B = data_1b[i]

        # Plot the lines in the current subplot
        axes[row, col].plot(row_A, label="Mult. Ch.")
        axes[row, col].plot(row_B, label="T/F")

        # Set the title for the current subplot
        axes[row, col].set_title(f"P={prompt_abbreviations[i]}")

        # Set the y-axis limits for the current subplot
        if i != 2:
            axes[row, col].set_ylim(0, 0.15)

        # Add the legend
        if i == 0:
            axes[row, col].legend()

    # Remove the last two empty subplots
    axes[2, 2].axis("off")
    axes[2, 3].axis("off")

    # Adjust the layout and display the plot
    plt.tight_layout()
    outfile_location = "figures/line_graph_1a_1b_comparison.png"
    plt.savefig(outfile_location)


def test_bonus():
    # For each prompt P, perform Welch’s t-test between the the populations Y(Q=mult. ch.) and Y(Q=T/F),
    # Q=0 denotes multiple choice question format, Q=1 denotes True/False question format
    # where each population consist of the ~1100 “correct” or “incorrect” ratings for the LLM
    #  (coded as 1 and 0 respectively).

    all_categories = np.array([count_categories_1a(), count_categories_1b()])
    
    correct_answers = all_categories [:, 1]
    incorrect_answers = all_categories [:, 2]

    for p_idx in range(correct_answers.shape[1]):
        population_a = np.concatenate((np.ones(sum(correct_answers[0, p_idx])), np.zeros(
            sum(incorrect_answers[0, p_idx]))))
        population_b = np.concatenate((np.ones(sum(correct_answers[1, p_idx])), np.zeros(
            sum(incorrect_answers[1, p_idx]))))
        ttest_result = stats.ttest_ind(
            population_a, population_b, equal_var=False)

        # Print the results of the two-sample Student's t-test for each prompt
        print(
            f"Two-sample Welch's t-test comparing the populations for two Q choices on prompt {p_idx+1}: statistic={ttest_result.statistic:.2f}, p-value={ttest_result.pvalue:.4f}")


if __name__ == "__main__":
    # make_comparisons_graph()
    # test_1b()
    test_bonus()
