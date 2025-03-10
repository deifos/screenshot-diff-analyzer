import streamlit as st
import os
from page_comparison_enhanced import analyze_differences

st.set_page_config(
    page_title="Image Comparison Analysis MVP",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 1rem;
    }
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        margin-bottom: 1rem;
    }
    .badge {
        background-color: #e0e7ff;
        color: #3730a3;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        display: inline-block;
    }
    .summary-section {
        margin-bottom: 1rem;
        padding: 0.75rem;
        border-radius: 0.5rem;
        background-color: #f8fafc;
    }
    .summary-item {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
        padding: 0.5rem;
        border-radius: 0.25rem;
        background-color: white;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .summary-icon {
        margin-right: 0.5rem;
        color: #3730a3;
        font-weight: bold;
    }
    .section-top { border-left: 4px solid #22c55e; }
    .section-middle { border-left: 4px solid #eab308; }
    .section-bottom { border-left: 4px solid #ef4444; }
    </style>
""", unsafe_allow_html=True)

# Header with updated title
st.title("Image Comparison Analysis MVP")
st.markdown("Upload and compare images to detect differences")

# Main content in tabs
tab1, tab2 = st.tabs(["Compare Images", "Settings"])

with tab1:
    # Create columns for image upload
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.subheader("Original Image")
        st.caption("Upload the base image for comparison")
        original_image = st.file_uploader("Upload original image", type=['png', 'jpg', 'jpeg', 'webp'], key="original_image")

    with right_col:
        st.subheader("Updated Image")
        st.caption("Upload the image to compare against")
        comparison_image = st.file_uploader("Upload comparison image", type=['png', 'jpg', 'jpeg', 'webp'], key="comparison_image")

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Analysis Settings")
    st.caption("Configure comparison parameters")
    
    threshold = st.slider(
        "Threshold",
        min_value=1,
        max_value=100,
        value=30,
        help="Lower values detect more subtle differences"
    )
    
    distance = st.slider(
        "Merge Distance",
        min_value=50,
        max_value=500,
        value=300,
        help="Maximum distance between differences to be merged"
    )
    
    generate_report = st.toggle(
        "Generate Detailed Report",
        value=True,
        help="Use AI to generate detailed descriptions of changes"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Process images when both are uploaded
if 'original_image' in st.session_state and 'comparison_image' in st.session_state:
    if st.session_state.original_image and st.session_state.comparison_image:
        # Save uploaded files temporarily
        temp_original = "temp_original.webp"
        temp_comparison = "temp_comparison.webp"
        
        with open(temp_original, "wb") as f:
            f.write(st.session_state.original_image.getbuffer())
        with open(temp_comparison, "wb") as f:
            f.write(st.session_state.comparison_image.getbuffer())

        try:
            # Run comparison
            with st.spinner("Analyzing differences..."):
                result = analyze_differences(
                    temp_original,
                    temp_comparison,
                    generate_report=generate_report,
                    threshold=threshold,
                    max_distance=distance
                )

            # Display results
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Analysis Results")
            st.caption("Differences found between images")
            
            # Create columns for image and report
            img_col, report_col = st.columns([2, 1])
            
            with img_col:
                if os.path.exists("differences_highlighted.webp"):
                    st.image("differences_highlighted.webp", use_container_width=True)
            
            with report_col:
                if "\n\n" in result:
                    summary, detailed = result.split("\n\n")
                    
                    # Format summary
                    differences = summary.split(". ")[0]
                    differences = differences.replace("Found ", "")
                    differences_list = differences.split(", ")
                    
                    st.subheader("Summary of Differences")
                    st.markdown('<div class="summary-section">', unsafe_allow_html=True)
                    
                    for diff in differences_list:
                        section = "top" if "top" in diff else "middle" if "middle" in diff else "bottom"
                        icon = "🔼" if section == "top" else "⏺️" if section == "middle" else "🔽"
                        st.markdown(
                            f'<div class="summary-item section-{section}">'
                            f'<span class="summary-icon">{icon}</span>{diff}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Format detailed report using expanders
                    if detailed:
                        st.markdown("### Detailed Changes")
                        detailed = detailed.replace("Detailed changes: ", "")
                        sections = detailed.split("In the ")
                        
                        for section in sections:
                            if section:
                                section_name = section.split(" section: ")[0]
                                section_content = section.split(" section: ")[1]
                                
                                with st.expander(f"{section_name.title()} Section"):
                                    differences = section_content.split("; ")
                                    for diff in differences:
                                        if diff.strip():
                                            st.markdown(f"- {diff.strip()}")
                else:
                    # Handle basic report with the same styling
                    st.subheader("Summary of Differences")
                    st.markdown('<div class="summary-section">', unsafe_allow_html=True)
                    
                    # Parse the basic report
                    result = result.replace("Found ", "")
                    result = result.split(". ")[0]  # Get only the differences part
                    differences_list = result.split(", ")
                    
                    for diff in differences_list:
                        section = "top" if "top" in diff else "middle" if "middle" in diff else "bottom"
                        icon = "🔼" if section == "top" else "⏺️" if section == "middle" else "🔽"
                        st.markdown(
                            f'<div class="summary-item section-{section}">'
                            f'<span class="summary-icon">{icon}</span>{diff}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    # Add a note about the output file
                    st.info("Check 'differences_highlighted.webp' for visual details.")
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during comparison: {str(e)}")
        finally:
            # Cleanup temporary files
            for temp_file in [temp_original, temp_comparison]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

else:
    st.info("Please upload both images to start comparison") 