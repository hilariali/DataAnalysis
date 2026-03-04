import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import re


def parse_input(raw_text: str) -> tuple[list[float], list[str]]:
    """Parse spreadsheet-pasted data into a list of floats.

    Splitting strategy:
    1. Split by newlines and tabs first (standard spreadsheet copy format).
    2. If that yields fewer than 2 tokens, fall back to splitting by
       semicolons and commas as well (CSV-style input).

    Within each token, commas are treated as thousands separators so that
    values like "1,234.56" are correctly parsed as 1234.56.
    Empty tokens and non-numeric entries are skipped with a warning.
    """
    # Primary split: newlines / carriage-returns / tabs
    tokens = re.split(r"[\n\r\t]+", raw_text.strip())

    # If that produced only one token, also try splitting by semicolons/commas
    if len(tokens) < 2:
        tokens = re.split(r"[;\s,]+", raw_text.strip())

    values: list[float] = []
    skipped: list[str] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        # Remove thousands separators (commas), currency symbols, percent signs
        cleaned = re.sub(r"[,$%]", "", token)
        try:
            values.append(float(cleaned))
        except ValueError:
            skipped.append(token)
    return values, skipped


def main() -> None:
    st.set_page_config(page_title="Normal Distribution Generator", page_icon="📊")

    st.title("📊 Normal Distribution Graph Generator")
    st.markdown(
        "Paste a column of numbers copied from a spreadsheet (Excel, Google Sheets, etc.) "
        "into the box below. The app will compute descriptive statistics and plot the "
        "distribution."
    )

    raw_input = st.text_area(
        label="Paste your data here (one value per line)",
        height=200,
        placeholder="12.5\n14.3\n11.8\n13.0\n...",
    )

    if not raw_input.strip():
        st.info("👆 Paste your data above to get started.")
        return

    values, skipped = parse_input(raw_input)

    if skipped:
        st.warning(
            f"⚠️ {len(skipped)} non-numeric value(s) were ignored: "
            + ", ".join(f'"{s}"' for s in skipped[:10])
            + ("..." if len(skipped) > 10 else "")
        )

    if len(values) < 2:
        st.error("❌ At least 2 numeric values are required to compute statistics.")
        return

    data = np.array(values)

    # --- Descriptive statistics ---
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1))          # sample standard deviation
    median = float(np.median(data))
    minimum = float(np.min(data))
    maximum = float(np.max(data))
    skewness = float(stats.skew(data))
    kurt = float(stats.kurtosis(data))          # excess kurtosis
    count = len(data)

    # Normality test (Shapiro-Wilk for n ≤ 5000, otherwise skipped)
    normality_note = ""
    if count <= 5000:
        stat_sw, p_sw = stats.shapiro(data)
        if p_sw > 0.05:
            normality_note = f"✅ Shapiro-Wilk test: p = {p_sw:.4f} (data appears normally distributed)"
        else:
            normality_note = f"⚠️ Shapiro-Wilk test: p = {p_sw:.4f} (data may not be normally distributed)"

    st.subheader("📋 Descriptive Statistics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Count", count)
    col2.metric("Mean", f"{mean:.4g}")
    col3.metric("Std Dev (s)", f"{std:.4g}")
    col4.metric("Median", f"{median:.4g}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Min", f"{minimum:.4g}")
    col6.metric("Max", f"{maximum:.4g}")
    col7.metric("Skewness", f"{skewness:.4f}")
    col8.metric("Excess Kurtosis", f"{kurt:.4f}")

    if normality_note:
        st.markdown(normality_note)

    # --- Plot ---
    st.subheader("📈 Distribution Plot")

    fig, ax = plt.subplots(figsize=(10, 5))

    # Histogram (density=True so we can overlay the PDF)
    n_bins = min(max(10, int(np.sqrt(count))), 50)
    ax.hist(data, bins=n_bins, density=True, alpha=0.6, color="steelblue",
            edgecolor="white", label="Data (histogram)")

    # Fitted normal distribution curve
    x = np.linspace(minimum - std, maximum + std, 400)
    pdf = stats.norm.pdf(x, loc=mean, scale=std)
    ax.plot(x, pdf, "r-", linewidth=2, label=f"Normal fit μ={mean:.4g}, σ={std:.4g}")

    # Reference lines
    ax.axvline(mean, color="red", linestyle="--", linewidth=1.2, alpha=0.8, label=f"Mean = {mean:.4g}")
    ax.axvline(mean - std, color="orange", linestyle=":", linewidth=1.2, alpha=0.8, label=f"±1σ = {mean - std:.4g}, {mean + std:.4g}")
    ax.axvline(mean + std, color="orange", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.axvline(mean - 2 * std, color="green", linestyle=":", linewidth=1.0, alpha=0.6, label=f"±2σ = {mean - 2*std:.4g}, {mean + 2*std:.4g}")
    ax.axvline(mean + 2 * std, color="green", linestyle=":", linewidth=1.0, alpha=0.6)

    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Normal Distribution", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    # 68-95-99.7 rule summary
    within_1s = np.sum((data >= mean - std) & (data <= mean + std)) / count * 100
    within_2s = np.sum((data >= mean - 2 * std) & (data <= mean + 2 * std)) / count * 100
    within_3s = np.sum((data >= mean - 3 * std) & (data <= mean + 3 * std)) / count * 100

    st.subheader("📐 Empirical Rule (68–95–99.7)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Within 1σ (exp. 68.3%)", f"{within_1s:.1f}%")
    c2.metric("Within 2σ (exp. 95.4%)", f"{within_2s:.1f}%")
    c3.metric("Within 3σ (exp. 99.7%)", f"{within_3s:.1f}%")


if __name__ == "__main__":
    main()
