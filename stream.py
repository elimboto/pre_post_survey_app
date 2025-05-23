import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
from scipy.stats import ttest_rel, wilcoxon
import openpyxl

#Display info on charts
textstr = 'Created at \nwww.tssfl.com'
plt.rcParams['xtick.labelsize'] = 16

class DataTransformer:
    def __init__(self, response_categories=None):
        self.response_categories = response_categories or [
            "Strongly Disagree", "Disagree", "Somewhat Agree", "Agree", "Strongly Agree"
        ]
        self.score_mapping = {
            "Strongly Disagree": 1,
            "Disagree": 2,
            "Somewhat Agree": 3,
            "Agree": 4,
            "Strongly Agree": 5
        }

    def transform_data(self, data, columns_to_process):
        target_columns = data[columns_to_process].copy()
        for col in target_columns.columns:
            target_columns[col] = target_columns[col].astype(str).str.strip()
            target_columns[col] = target_columns[col].str.replace(r'\s+', ' ', regex=True)

        results = []
        for col in target_columns.columns:
            scores = target_columns[col].map(self.score_mapping)
            mean_score = scores.mean()

            response_counts = target_columns[col].value_counts(normalize=True) * 100
            response_percentages = {resp: response_counts.get(resp, 0) for resp in self.response_categories}
            response_percentages["Mean Score"] = mean_score
            response_percentages["Question"] = col
            results.append(response_percentages)

        return pd.DataFrame(results, columns=["Question"] + self.response_categories + ["Mean Score"])

    def visualize_mean_scores(self, pre_transformed, post_transformed, color_pre, color_post, chart_type="Bar"):
        questions = pre_transformed["Question"].tolist()
        pre_mean = pre_transformed["Mean Score"].tolist()
        post_mean = post_transformed["Mean Score"].tolist()
        fontsize = 16
        assert questions == post_transformed["Question"].tolist(), "Questions in Pre and Post datasets do not match."

        fig, ax = plt.subplots(figsize=(12, 8 + len(questions) * 0.35))
        y_positions = np.arange(len(questions))
        bar_width = 0.4
        offset = 0.25 / 2.54
        text_color = 'green'

        if chart_type == "Bar":
            pre_bars = ax.barh(y_positions + bar_width / 2, pre_mean, bar_width, label="Pre-Intervention", color=color_pre)
            post_bars = ax.barh(y_positions - bar_width / 2, post_mean, bar_width, label="Post-Intervention", color=color_post)

            for bar, mean in zip(pre_bars, pre_mean):
                ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2, f"{mean:.2f}", va="center", ha="left", color=text_color, fontsize=fontsize)
            for bar, mean in zip(post_bars, post_mean):
                ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2, f"{mean:.2f}", va="center", ha="left", color=text_color, fontsize=fontsize)

            questions = sorted(questions, key=lambda x: int(x[1:]), reverse=True)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(questions, fontsize=fontsize)
            
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, ncol=1, loc="lower left", bbox_to_anchor=(0, 1.06, 0.24, 0.08),
                  borderaxespad=0, mode="expand", fontsize=fontsize)
        else:
            ax.plot(questions, pre_mean, marker='o', label='Pre-Intervention', color=color_pre)
            ax.plot(questions, post_mean, marker='o', label='Post-Intervention', color=color_post)
            for i, (pm, qm) in enumerate(zip(pre_mean, post_mean)):
                ax.text(i, pm, f"{pm:.2f}", va="bottom", ha="center", fontsize=fontsize-2, color=color_pre)
                ax.text(i, qm, f"{qm:.2f}", va="top", ha="center", fontsize=fontsize-2, color=color_post)
            ax.set_xticks(range(len(questions)))
            ax.set_xticklabels(questions, rotation=45, ha="right", fontsize=fontsize)

        ax.set_xlabel("Mean Score (scale 1 to 5)", fontsize=fontsize)
        ax.set_title("Comparison of Mean Scores (Pre- vs. Post-Intervention)", fontsize=fontsize)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.06), fontsize=fontsize)

        for spine in ['right', 'left', 'top']:
            ax.spines[spine].set_visible(False)

        plt.gcf().text(0.8, 0.94, textstr, fontsize=14, color='green')
        plt.tight_layout()
        st.pyplot(fig)

        buf = BytesIO()
        plt.savefig(buf, format="png")
        st.download_button("Download Chart as Image", buf.getvalue(), "chart.png", mime="image/png")

    def create_table(self, transformed_data):
        df = transformed_data.set_index("Question") #Transform table using .T
        df.index.name = "Response (%) | Mean Score (1–5)"
        return df

def sanitize_sheet_name(name):
    return ''.join(c for c in name if c not in r'[]:*?/\\').strip()[:31]

def convert_df_to_excel(df_dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for name, df in df_dict.items():
            safe_name = sanitize_sheet_name(name)
            df.to_excel(writer, sheet_name=safe_name, index=False)
    output.seek(0)
    return output

def generate_ai_insights(pre_df, post_df):
    deltas = post_df["Mean Score"] - pre_df["Mean Score"]
    improved = deltas[deltas > 0]
    declined = deltas[deltas < 0]
    unchanged = deltas[deltas == 0]

    summary = f"""
    **AI-Assisted Insights:**

    - {len(improved)} question(s) showed improvement.
    - {len(declined)} question(s) declined.
    - {len(unchanged)} question(s) remained unchanged.

    **Improved:** {', '.join(improved.index)}
    **Declined:** {', '.join(declined.index)}
    **Unchanged:** {', '.join(unchanged.index)}

    Greatest positive change: **{deltas.max():.2f}** in *{deltas.idxmax()}*.
    Greatest negative change: **{deltas.min():.2f}** in *{deltas.idxmin()}*.
    """
    return summary

def run_statistical_tests(pre_df, post_df):
    paired_data = pd.DataFrame({
        "pre": pre_df["Mean Score"],
        "post": post_df["Mean Score"]
    }).dropna()

    if paired_data.empty:
        return "**Statistical Significance Test:**\n\n- Not enough valid data for statistical testing."

    try:
        # Attempt Wilcoxon, handle ties with exact=False for better robustness
        stat, p = wilcoxon(paired_data["pre"], paired_data["post"], alternative='two-sided', exact=False)  #Two sided test
        test_name = "Wilcoxon Signed-Rank Test"
        #Check for paired data - if there is a mismatch of respondents, the paired test is wrong.
        if len(pre_df) != len(post_df):
            test_name += " (Note: Data may not be truly paired. Consider independent samples test.)"
    except Exception as e:
        print(f"Wilcoxon failed: {e}")  # Log the error for debugging
        try:
            #Try paired t-test if Wilcoxon fails - but again, check for paired data
            stat, p = ttest_rel(paired_data["pre"], paired_data["post"])
            test_name = "Paired t-Test"
            if len(pre_df) != len(post_df):
                test_name += " (Note: Data may not be truly paired. Consider independent samples test.)"
        except Exception as e:
            print(f"Paired t-test failed: {e}")
            return "**Statistical Significance Test:**\n\n- Data unsuitable for both paired tests."


    result = f"""
    **Statistical Significance Test:**
    - Test used: {test_name}
    - Test Statistic: {stat:.4f}
    - p-value: {p:.4f}
    {'✅ Statistically significant (p < 0.05)' if p < 0.05 else '❌ Not statistically significant (p ≥ 0.05)'}
    """
    return result


def run_statistical_tests(pre_df, post_df):
    paired_data = pd.DataFrame({
        "pre": pre_df["Mean Score"],
        "post": post_df["Mean Score"]
    }).dropna()

    if paired_data.empty:
        return "**Statistical Significance Test:**\n\n- Not enough valid data for statistical testing."

    try:
        stat, p = wilcoxon(paired_data["pre"], paired_data["post"])
        test_name = "Wilcoxon Signed-Rank Test"
    except Exception:
        stat, p = ttest_rel(paired_data["pre"], paired_data["post"])
        test_name = "Paired t-Test"

    result = f"""
    **Statistical Significance Test:**

    - Test used: {test_name}
    - Test Statistic: {stat:.4f}
    - p-value: {p:.4f}

    {'✅ Statistically significant (p < 0.05)' if p < 0.05 else '❌ Not statistically significant (p ≥ 0.05)'}
    """
    return result

def main():
    st.title("📊 Survey Insights Generator, Developed by TSSFL Team")
    st.markdown("""
    This app processes pre/post survey data, calculates summary statistics, generates comparative charts, performs statistical tests,
    and provides AI-assisted insights.
    """)

    uploaded_pre = st.file_uploader("Upload Pre-Intervention Survey (Excel or CSV)", type=["xlsx", "xls", "csv"], key="pre")
    uploaded_post = st.file_uploader("Upload Post-Intervention Survey (Excel or CSV)", type=["xlsx", "xls", "csv"], key="post")

    if uploaded_pre and uploaded_post:
        pre_data = pd.read_csv(uploaded_pre) if uploaded_pre.name.endswith("csv") else pd.read_excel(uploaded_pre)
        post_data = pd.read_csv(uploaded_post) if uploaded_post.name.endswith("csv") else pd.read_excel(uploaded_post)

        st.subheader("Select Columns to Process")
        common_columns = list(set(pre_data.columns).intersection(post_data.columns))
        selected_columns = st.multiselect("Select Likert-scale Question Columns", common_columns)

        if selected_columns:
            dt = DataTransformer()
            pre_transformed = dt.transform_data(pre_data, selected_columns)
            post_transformed = dt.transform_data(post_data, selected_columns)

            st.subheader("📈 Chart Type")
            chart_type = st.radio("Select chart type", ["Bar", "Line"], horizontal=True)
            color_pre = st.color_picker("Pick color for Pre-Intervention", "#f4d641")
            color_post = st.color_picker("Pick color for Post-Intervention", "#4286f4")
            dt.visualize_mean_scores(pre_transformed, post_transformed, color_pre, color_post, chart_type)

            st.subheader("📋 Tabular Summary")
            st.write("Pre-Intervention Summary")
            st.dataframe(dt.create_table(pre_transformed))
            st.write("Post-Intervention Summary")
            st.dataframe(dt.create_table(post_transformed))

            excel_data = convert_df_to_excel({
                "Pre-Intervention": pre_transformed,
                "Post-Intervention": post_transformed
            })
            st.download_button("Download Summary Excel", excel_data.getvalue(), "summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            st.subheader("🧠 AI Insights")
            st.markdown(generate_ai_insights(pre_transformed.set_index("Question"), post_transformed.set_index("Question")))

            st.subheader("📐 Statistical Testing")
            st.markdown(run_statistical_tests(pre_transformed.set_index("Question"), post_transformed.set_index("Question")))

if __name__ == "__main__":
    main()

