import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def render_conversation_messages(conv_data):
    """
    Renders a conversation thread in a Streamlit interface with expandable view.
    This function takes conversation data and displays it in a structured format using Streamlit's
    chat message components. Each message is displayed according to its source (user or agent) with
    appropriate styling. For agent messages, any tool invocations are also displayed with their
    status before the actual message content.
    Args:
        conv_data (dict): A dictionary containing conversation data with the following structure:
            {
                "messages": [
                    {
                        "source": str,  # "user" or "agent"
                        "message": str,  # The actual message content
                        "tool_invocations": [  # Optional, only for agent messages
                            {
                                "invocation": str,  # Tool invocation command
                                "status": str       # Status of the tool invocation
                            }
                        ]
                    }
                ]
            }
    Returns:
        None: The function directly renders to the Streamlit interface
    Example:
        conv_data = {
            "messages": [
                {"source": "user", "message": "Hello"},
                {"source": "agent", "message": "Hi", "tool_invocations": [{"invocation": "greet", "status": "success"}]}
            ]
        }
        render_conversation_messages(conv_data)
    """
    # Display the messages
    with st.expander("View Full Conversation"):
        for idx, msg in enumerate(conv_data.get("messages", [])):
            source = msg.get("source", "unknown")
            message = msg.get("message", "")
            
            if source.lower() == "user":
                with st.chat_message("user"):
                    st.markdown(message)
            elif source.lower() == "agent":
                with st.chat_message("assistant"):
                    tool_invocations = msg.get("tool_invocations", [])
                    if tool_invocations:
                        for tool in tool_invocations:
                            st.markdown(f"**Tool Invocation:** `{tool.get('invocation', '')}` _(**Status:** {tool.get('status', '')})_")
                        st.markdown("---")
                    st.markdown(message)

def show_top_level_metrics(all_segments, selected_segments, filtered_conversations, filtered_accuracy):
    """
    Display top-level metrics in a Streamlit dashboard using columns layout.
    This function creates 5 metric cards showing key statistics about instruction following accuracy
    and conversation data.
    Parameters:
        all_segments (list): List of all available segments in the dataset
        selected_segments (list): List of segments currently selected for analysis
        filtered_conversations (list): List of conversation dictionaries matching current filters
        filtered_accuracy (float): The calculated IF accuracy percentage for filtered conversations
    Each metric displayed:
    - Overall IF Accuracy: Shows the instruction following accuracy as a percentage
    - Total Conversations: Count of conversations in the filtered dataset
    - Total Applicable Instructions: Sum of applicable instructions across filtered conversations
    - Total Followed Instructions: Sum of followed instructions across filtered conversations 
    - Selected Segments: Shows ratio of selected segments to total segments
    Notes:
        - Uses Streamlit's column and metric components for layout
        - Handles empty filtered_conversations case by showing '0' for instruction metrics
    """
    # Top-level metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Overall IF Accuracy", f"{filtered_accuracy:.1f}%")
    
    with col2:
        total_conversations = len(filtered_conversations)
        st.metric("Total Conversations", total_conversations)
    
    with col3:
        if filtered_conversations:
            tot_applicable_instructions = sum(conv["conversation_if_accuracy"]["applicable_instructions"] 
                                           for conv in filtered_conversations)
            st.metric("Total Applicable Instructions", f"{tot_applicable_instructions}")
        else:
            st.metric("Total Applicable Instructions", "0")
    
    with col4:
        if filtered_conversations:
            tot_followed_instructions = sum(conv["conversation_if_accuracy"]["followed_instructions"] 
                                          for conv in filtered_conversations)
            st.metric("Total Followed Instructions", f"{tot_followed_instructions}")
        else:
            st.metric("Total Followed Instructions", "0")

    with col5:
        st.metric("Selected Segments", f"{len(selected_segments)} of {len(all_segments)}")


def show_segment_analysis(conv_df):
    """
    Creates and displays segment-level analysis visualizations for IF (Instruction Following) accuracy.
    This function processes conversation data grouped by segments to show:
    1. A bar chart of IF accuracy percentages across different segments
    2. A detailed dataframe with segment-specific metrics
    Parameters:
    ----------
    conv_df : pandas.DataFrame
        Input DataFrame containing conversation data with columns:
        - 'Segment'
        - 'Conversation ID'
        - 'Applicable Instructions'
        - 'Followed Instructions'
    Returns:
    -------
    None
        Displays visualizations directly using Streamlit (st):
        - Bar chart showing IF accuracy by segment
        - Dataframe with detailed segment metrics
    Notes:
    -----
    The function calculates IF accuracy as: (Followed Instructions / Applicable Instructions) * 100
    Uses Plotly Express for visualization with a RdYlGn (Red-Yellow-Green) color scale
    """
    # Segment Analysis tab contents
    st.subheader("Segment-Level IF Accuracy")
    
    # Group by segment
    segment_df = conv_df.groupby("Segment").agg(
        conversation_count=("Conversation ID", "count"),
        total_applicable=("Applicable Instructions", "sum"),
        total_followed=("Followed Instructions", "sum")
    ).reset_index()
    
    # Add calculated IF accuracy from totals
    segment_df["calculated_accuracy"] = segment_df["total_followed"] / segment_df["total_applicable"] * 100
    
    # Create segment bar chart
    fig = px.bar(segment_df, x="Segment", y="calculated_accuracy",
                hover_data=["conversation_count", "total_applicable", "total_followed"],
                color="calculated_accuracy", color_continuous_scale="RdYlGn",
                range_color=[0, 100], height=400,
                labels={"calculated_accuracy": "IF Accuracy (%)"})
    fig.update_layout(xaxis_title="Segment", yaxis_title="IF Accuracy (%)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Segment Details")
    st.dataframe(
        segment_df.rename(columns={
            "conversation_count": "Conversation Count",
            "total_applicable": "Total Applicable Instructions",
            "total_followed": "Total Followed Instructions",
            "calculated_accuracy": "IF Accuracy (%)"
        }),
        hide_index=True
    )


def show_instruction_analysis(filtered_conversations, conversation_map):
    """
    Displays and analyzes instruction following accuracy data in a Streamlit dashboard.
    This function creates an interactive visualization and analysis of instruction following
    metrics from conversation data. It includes:
    - A horizontal bar chart showing success rates for different instructions
    - Detailed diagnostic metrics for selected instructions
    - Analysis of failed conversations with their full context
    Parameters:
        filtered_conversations (list): List of conversation dictionaries containing instruction
            following analyses and related metadata. Each conversation should have:
            - conversation_id
            - instruction_following_analyses
            - segment (optional)
        conversation_map (dict): Dictionary mapping conversation IDs to their full conversation
            data for detailed analysis of failures
    Returns:
        None - Renders visualizations directly to the Streamlit dashboard
    """
    st.markdown("## Instruction Analysis")

    # Create a list to store all atomic instructions and their metrics
    atomic_instructions_data = []
    for conv in filtered_conversations:
        for instr in conv["instruction_following_analyses"]:
            if instr.get("isApplicable", False):
                # Extract index for display
                display_index = instr.get("index", "") if instr.get("type", "").startswith("workflow") else ""
                display_prefix = f"[{display_index}] " if display_index != "" else ""
                
                atomic_instructions_data.append({
                    "instruction": instr["instruction"],
                    "display_instruction": f"{display_prefix}{instr['instruction']}",
                    "followed": instr.get("isFollowed", False),
                    "conversation_id": conv["conversation_id"],
                    "segment": conv.get("segment", "Unknown"),
                    "metrics": instr.get("diagnosticMetrics", {}),
                    "rationale": instr.get("rationale", ""),
                    # Add type and index for sorting
                    "type": instr.get("type", "generic"),
                    "index": instr.get("index", float('inf'))  # Use infinity for generic instructions
                })

    # Create DataFrame and sort before grouping
    atomic_df = pd.DataFrame(atomic_instructions_data)
    
    # Sort by type (workflow first) and index within workflow
    atomic_df["is_workflow"] = atomic_df["type"].apply(lambda x: 0 if x.startswith("workflow") else 1)
    atomic_df["workflow_name"] = atomic_df["type"].apply(lambda x: x.split(" - ")[1] if x.startswith("workflow") and " - " in x else "zzzz")
    
    # Sort the dataframe before grouping by instruction
    sorted_atomic_df = atomic_df.sort_values(by=["is_workflow", "workflow_name", "index"])
    
    # Use the sorted dataframe for grouping
    instruction_summary = sorted_atomic_df.groupby("display_instruction").agg(
        applicable_count=("display_instruction", "count"),
        followed_count=("followed", "sum"),
        original_instruction=("instruction", "first")  # Keep original for reference
    ).reset_index()
    
    # Preserve the original sorting order from sorted_atomic_df
    instruction_order = sorted_atomic_df.drop_duplicates("display_instruction")[["display_instruction"]].reset_index(drop=True)
    instruction_summary = instruction_order.merge(instruction_summary, on="display_instruction")
    
    instruction_summary["success_rate"] = (instruction_summary["followed_count"] / instruction_summary["applicable_count"] * 100)

    # Reverse the order of instructions for the plot by using iloc[::-1]
    plot_data = instruction_summary.iloc[::-1].copy()
    
    # Plot instruction success rates with horizontal bars
    fig = px.bar(plot_data, y="display_instruction", x="success_rate",
            orientation='h',  # Makes bars horizontal
            color="success_rate",
            color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            height=400,
            labels={"success_rate": "Instruction Following Accuracy (%)",
                "display_instruction": "Instruction"})
    st.plotly_chart(fig, use_container_width=True)

    # Instruction selector
    selected_display_instruction = st.selectbox(
        "Select an instruction to analyze:",
        instruction_summary["display_instruction"].tolist()
    )

    # Get the original instruction to filter data
    selected_instruction = instruction_summary[instruction_summary["display_instruction"] == selected_display_instruction]["original_instruction"].iloc[0]

    # Filter data for selected instruction
    selected_instr_data = atomic_df[atomic_df["instruction"] == selected_instruction]

    # Aggregate diagnostic metrics
    metrics_summary = {}
    for _, row in selected_instr_data.iterrows():
        for metric, data in row["metrics"].items():
            if metric not in metrics_summary:
                metrics_summary[metric] = {"applicable": 0, "followed": 0}
            if data.get("isApplicable", False):
                metrics_summary[metric]["applicable"] += 1
                if data.get("isFollowed", False):
                    metrics_summary[metric]["followed"] += 1

    # Create metrics visualization
    metrics_long = []
    for metric, data in metrics_summary.items():
        metrics_long.extend([
        {"metric": metric, "type": "Applicable", "count": data["applicable"]},
        {"metric": metric, "type": "Followed", "count": data["followed"]}
        ])

    metrics_df = pd.DataFrame(metrics_long)

    if not metrics_df.empty:
        fig = px.bar(metrics_df, x="metric", y="count", color="type",
            title=f"Diagnostic Metrics for: {selected_instruction}",
            barmode="group",
            labels={"count": "Count", "metric": "Metric", "type": "Type"})
        fig.update_yaxes(dtick=1)  # Set interval between ticks to 1
        st.plotly_chart(fig, use_container_width=True)

    # Show failures
    failed_conversations = selected_instr_data[~selected_instr_data["followed"]]
    if not failed_conversations.empty:
        failed_conv_id = st.selectbox(
            "Select a failed conversation to analyze:",
            failed_conversations["conversation_id"].tolist()
        )
        
        # Get conversation details
        conv_data = conversation_map.get(failed_conv_id)
            
        render_conversation_messages(conv_data)
            
        # Show diagnostic metrics for this instance
        st.markdown("### Diagnostic Metrics Analysis")
        failed_instance = selected_instr_data[selected_instr_data["conversation_id"] == failed_conv_id].iloc[0]
        for metric, data in failed_instance["metrics"].items():
            if data.get("isApplicable", False):
                status = "✅" if data.get("isFollowed", False) else "❌"
                st.markdown(f"**{metric}**: {status}")
                st.markdown(f"Rationale: {data.get('rationale', 'No rationale provided')}")


def show_diagnostic_metrics_analysis(filtered_conversations, conversation_map):
    """
    Analyze and display diagnostic metrics for conversations in a Streamlit dashboard.
    This function processes conversation data to analyze various diagnostic metrics (interactivity,
    contextRetention, reasoning, toolCall, responseFormat) and presents the results through
    interactive visualizations and drilldown capabilities.
    Parameters:
        filtered_conversations (list): List of conversation dictionaries containing instruction
            following analyses and diagnostic metrics data.
        conversation_map (dict): Dictionary mapping conversation IDs to full conversation data
            for detailed viewing.
    The function provides:
    - Aggregate success rates across all diagnostic metrics
    - Interactive bar charts showing metric performance
    - Segment-wise breakdown of metric performance
    - Detailed drilldown capability for each metric including:
        - Success rate statistics
        - Segment-wise performance visualization
        - List of all applicable conversations
        - Detailed conversation view with specific metric issues
    The display includes:
    - Overview bar chart of success rates
    - Metric selection for detailed analysis
    - Segment performance breakdown
    - Individual conversation examination
    - Color-coded conversation tables (green for success, red for failure)
    Requirements:
        - Streamlit (st)
        - Pandas (pd)
        - Plotly Express (px)
    Note: This function is designed to be used within a Streamlit application and relies on
    Streamlit's display components for visualization.
    """
    # DIAGNOSTIC METRICS AS PRIMARY DRILLDOWN
    st.markdown("## Diagnostic Metrics Analysis")
    
    # Collect and analyze diagnostic metrics across all filtered conversations
    all_diagnostic_details = []
    all_metrics = {
        "interactivity": {"applicable": 0, "followed": 0, "details": []},
        "contextRetention": {"applicable": 0, "followed": 0, "details": []},
        "reasoning": {"applicable": 0, "followed": 0, "details": []},
        "toolCall": {"applicable": 0, "followed": 0, "details": []},
        "responseFormat": {"applicable": 0, "followed": 0, "details": []}
    }
    
    # Collect all metrics data
    for conv in filtered_conversations:
        conv_id = conv["conversation_id"]
        segment = conv.get("segment", "Unknown")
        
        for instr in conv["instruction_following_analyses"]:
            if instr.get("isApplicable", False) and "diagnosticMetrics" in instr:
                # Add index prefix
                display_index = instr.get("index", "") if instr.get("type", "").startswith("workflow") else ""
                display_prefix = f"[{display_index}] " if display_index != "" else ""
                instruction_text = f"{display_prefix}{instr['instruction']}"
                instruction_followed = instr.get("isFollowed", False)
                
                metrics = instr["diagnosticMetrics"]
                for metric_name, metric_data in metrics.items():
                    if metric_data.get("isApplicable", False):
                        is_followed = metric_data.get("isFollowed", False)
                        rationale = metric_data.get("rationale", "No rationale provided")
                        
                        # Add to aggregate counts
                        all_metrics[metric_name]["applicable"] += 1
                        if is_followed:
                            all_metrics[metric_name]["followed"] += 1
                        
                        # Add details for deeper analysis
                        all_metrics[metric_name]["details"].append({
                            "Conversation ID": conv_id,
                            "Segment": segment,
                            "Instruction": instruction_text,
                            "Instruction Followed": instruction_followed,
                            "Metric Followed": is_followed,
                            "Rationale": rationale
                        })
                        
                        all_diagnostic_details.append({
                            "Metric": metric_name,
                            "Conversation ID": conv_id,
                            "Segment": segment,
                            "Instruction": instruction_text,
                            "Instruction Followed": instruction_followed,
                            "Metric Followed": is_followed,
                            "Rationale": rationale
                        })
    
    # Create aggregate metrics dataframe
    agg_metrics_data = []
    for metric_name, metric_info in all_metrics.items():
        if metric_info["applicable"] > 0:
            agg_metrics_data.append({
                "Metric": metric_name,
                "Applicable Count": metric_info["applicable"],
                "Followed Count": metric_info["followed"],
                "Success Rate": round(metric_info["followed"] / metric_info["applicable"] * 100, 1) if metric_info["applicable"] > 0 else 0
            })
    
    agg_metrics_df = pd.DataFrame(agg_metrics_data)
    
    if not agg_metrics_df.empty:
        # Overview of all metrics
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Create bar chart for success rates
            fig = px.bar(agg_metrics_df, x="Metric", y="Success Rate",
                        color="Success Rate", color_continuous_scale="RdYlGn",
                        range_color=[0, 100], height=400)
            fig.update_layout(xaxis_title="Diagnostic Metric", yaxis_title="Success Rate (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Show metrics table
            st.dataframe(agg_metrics_df)
        
        # DRILLDOWN: Select a specific diagnostic metric to analyze
        if agg_metrics_data:
            selected_metric = st.selectbox(
                "Select a diagnostic metric to analyze:",
                [metric["Metric"] for metric in agg_metrics_data]
            )
            
            # Get details for selected metric
            metric_details = all_metrics[selected_metric]["details"]
            metric_details_df = pd.DataFrame(metric_details)
            
            # Analyze selected metric
            st.markdown(f"### Analysis of {selected_metric}")
            
            # Summary statistics
            metric_info = all_metrics[selected_metric]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                success_rate = (metric_info["followed"] / metric_info["applicable"] * 100) if metric_info["applicable"] > 0 else 0
                st.metric(f"{selected_metric} Success Rate", f"{success_rate:.1f}%")
            
            with col2:
                st.metric("Applicable Count", metric_info["applicable"])
            
            with col3:
                st.metric("Followed Count", metric_info["followed"])
            
            # Segment breakdown for this metric
            if not metric_details_df.empty:
                segment_metric_df = metric_details_df.groupby("Segment").agg(
                    applicable=("Metric Followed", "count"),
                    followed=("Metric Followed", "sum")
                ).reset_index()
                
                segment_metric_df["success_rate"] = segment_metric_df["followed"] / segment_metric_df["applicable"] * 100
                
                # Segment performance for this metric
                st.markdown(f"#### {selected_metric} Performance by Segment")
                
                fig = px.bar(segment_metric_df, x="Segment", y="success_rate",
                            hover_data=["applicable", "followed"],
                            color="success_rate", color_continuous_scale="RdYlGn",
                            range_color=[0, 100], height=300,
                            labels={"success_rate": f"{selected_metric} Success Rate (%)"})
                st.plotly_chart(fig, use_container_width=True)
                
                # All conversations where this metric was applicable
                st.markdown(f"#### All Conversations with {selected_metric} Metrics")
                
                # Sort by metric followed status (False first) and then by segment
                all_applicable_df = metric_details_df.sort_values(by=["Metric Followed", "Segment", "Conversation ID"])
                
                # Show table with color coding based on metric followed status
                st.dataframe(
                    all_applicable_df[["Conversation ID", "Segment", "Instruction", "Metric Followed"]].style.apply(
                        lambda x: ['background-color: #d4f1dd' if x['Metric Followed'] else 'background-color: #f1d4d4' 
                                for i in range(len(x))], axis=1
                    ),
                    height=300
                )
                
                # Show conversation details for a selected conversation
                st.markdown("##### Conversation Details")
                
                # Use unique conversation IDs from applicable metrics
                unique_conv_ids = all_applicable_df["Conversation ID"].unique()
                
                if len(unique_conv_ids) > 0:
                    selected_conv_id = st.selectbox(
                        f"Select a conversation to examine {selected_metric} details:",
                        unique_conv_ids
                    )
                    
                    # Get conversation data and metrics data
                    conv_metric_table = all_applicable_df[all_applicable_df["Conversation ID"] == selected_conv_id]
                    
                    if not conv_metric_table.empty:
                        
                        # Get the full conversation if available
                        if selected_conv_id in conversation_map:
                            conv_data = conversation_map[selected_conv_id]
                            render_conversation_messages(conv_data)
                        else:
                            st.warning("Full conversation details not available.")
                        
                        # Display metric issues for this conversation
                        st.markdown(f"###### \"{selected_metric}\" Issues in this Conversation")
                        
                        # Check if there are any failures
                        has_failures = any(not row["Metric Followed"] for _, row in conv_metric_table.iterrows())
                        
                        if has_failures:
                            for _, instance in conv_metric_table.iterrows():
                                if instance["Metric Followed"] == False:
                                    with st.container(border=True):
                                        st.markdown(f"**Instruction:** {instance['Instruction']}")
                                        st.markdown(f"**Rationale:** {instance['Rationale']}")
                        else:
                            st.success(f"No issues found for {selected_metric} in this conversation!")
                else:
                    st.info(f"No conversations found with applicable {selected_metric} metrics.")
            else:
                st.info(f"No data available for {selected_metric}")
        else:
            st.info("No diagnostic metrics data available")
    else:
        st.info("No diagnostic metrics data available in the selected conversations")


def show_conversation_analysis(conv_df, filtered_conversations, conversation_map):
    """
    Displays and analyzes conversation-level instruction following accuracy data in a Streamlit dashboard.
    This function creates an interactive dashboard that shows:
    1. A bar chart of IF (Instruction Following) accuracy across all conversations
    2. Detailed analysis of individual conversations including:
        - Full conversation messages
        - Instruction-level analysis with color-coded success/failure
        - Diagnostic metrics for specific instructions
    Parameters:
         conv_df (pd.DataFrame): DataFrame containing conversation-level metrics including:
              - Conversation ID
              - Segment
              - IF Accuracy
              - Applicable Instructions
              - Followed Instructions
         filtered_conversations (list): List of dictionaries containing detailed conversation data including:
              - conversation_id
              - segment
              - instruction_following_analyses
         conversation_map (dict): Dictionary mapping conversation IDs to their full message history
    Returns:
         None - Displays results directly in Streamlit dashboard
    Notes:
         - Uses plotly for interactive visualizations
         - Employs color coding (green/red) to indicate instruction following success/failure
         - Provides drill-down capability from conversation to instruction to diagnostic metrics level
    """
    # Conversation Analysis tab contents
    st.markdown("## Conversation-Level IF Accuracy")
    
    # Sort conversations by segment and then ID
    sorted_conv_df = conv_df.sort_values(by=["Segment", "Conversation ID"])
    
    fig = px.bar(sorted_conv_df, x="Conversation ID", y="IF Accuracy", 
                hover_data=["Segment", "Applicable Instructions", "Followed Instructions"],
                color="IF Accuracy", color_continuous_scale="RdYlGn",
                range_color=[0, 100], height=400)
    fig.update_layout(xaxis_title="Conversation ID", yaxis_title="IF Accuracy (%)")
    st.plotly_chart(fig, use_container_width=True)
    
    
    # Conversation selector for drilldown
    selected_conversation = st.selectbox(
        "Select a conversation to analyze:",
        [f"{conv['conversation_id']} ({conv.get('segment', 'Unknown')})" for conv in filtered_conversations]
    )
    
    # Extract conversation ID from selection
    selected_conv_id = selected_conversation.split(" (")[0]
    
    # Get selected conversation data
    selected_conv_data = next(conv for conv in filtered_conversations 
                            if conv["conversation_id"] == selected_conv_id)
    
    # Display the full conversation if available
    if selected_conv_id in conversation_map:
        conv_data = conversation_map[selected_conv_id]
        
        st.markdown("### Conversation Messages")
        render_conversation_messages(conv_data)
    
    # Create dataframe for instructions
    instruction_data = []
    for instr in selected_conv_data["instruction_following_analyses"]:
        if instr.get("isApplicable", False):
            # Extract index for display
            display_index = instr.get("index", "") if instr.get("type", "").startswith("workflow") else ""
            display_prefix = f"[{display_index}] " if display_index != "" else ""
            
            instruction_data.append({
                "Instruction": f"{display_prefix}{instr['instruction']}",
                "Original Instruction": instr["instruction"],
                "Type": instr["type"],
                "Is Followed": instr.get("isFollowed", False),
                "Expected Tool": instr.get("expected_tool", "None"),
                "Rationale": instr.get("rationale", ""),
                # Add these fields for sorting
                "is_workflow": 0 if instr["type"].startswith("workflow") else 1,
                "workflow_name": instr["type"].split(" - ")[1] if instr["type"].startswith("workflow") and " - " in instr["type"] else "zzzz",
                "index": instr.get("index", float('inf'))  # Use infinity for generic instructions
            })

    instr_df = pd.DataFrame(instruction_data)
    # Sort by workflow first, then by workflow name, then by index
    instr_df = instr_df.sort_values(by=["is_workflow", "workflow_name", "index"])

    # Display instruction table with color coding based on followed status
    st.markdown("### Instructions")
    st.dataframe(
        instr_df[["Instruction", "Type", "Is Followed", "Expected Tool"]].style.apply(
            lambda x: ['background-color: #d4f1dd' if x['Is Followed'] else 'background-color: #f1d4d4' 
                    for i in range(len(x))], axis=1
        ),
        height=300
    )

    # Select an instruction to view its diagnostic metrics
    applicable_instructions = [instr for instr in selected_conv_data["instruction_following_analyses"] 
                            if instr.get("isApplicable", False)]

    if applicable_instructions:
        # Sort instructions for the dropdown
        sorted_instructions = sorted(
            applicable_instructions, 
            key=lambda x: (
                0 if x.get("type", "").startswith("workflow") else 1,  # workflows first
                x.get("type", "").split(" - ")[1] if x.get("type", "").startswith("workflow") and " - " in x.get("type", "") else "zzzz",  # workflow name
                x.get("index", float('inf'))  # index within workflow
            )
        )
        
        # Add index to each instruction for display
        display_instructions = []
        for instr in sorted_instructions:
            display_index = instr.get("index", "") if instr.get("type", "").startswith("workflow") else ""
            display_prefix = f"[{display_index}] " if display_index != "" else ""
            display_instructions.append({
                "display_text": f"{display_prefix}{instr['instruction']}",
                "original_instruction": instr["instruction"]
            })
        
        selected_display_instruction = st.selectbox(
            "Select an instruction to view diagnostic metrics:",
            [item["display_text"] for item in display_instructions],
            key="conv_tab_instruction_select" 
        )
        
        # Find the original instruction from the selected display text
        selected_instruction = next(item["original_instruction"] for item in display_instructions 
                                if item["display_text"] == selected_display_instruction)
        
        # Get selected instruction data
        selected_instr_data = next(instr for instr in applicable_instructions 
                                if instr["instruction"] == selected_instruction)
        
        # Check if diagnostic metrics exist
        if "diagnosticMetrics" in selected_instr_data:
            metrics = selected_instr_data["diagnosticMetrics"]
            
            # Create metrics dataframe
            metrics_data = []
            for metric_name, metric_data in metrics.items():
                if metric_data.get("isApplicable", False):
                    metrics_data.append({
                        "Metric": metric_name,
                        "Is Followed": metric_data.get("isFollowed", False),
                        "Rationale": metric_data.get("rationale", "")
                    })
                else:
                    metrics_data.append({
                        "Metric": metric_name,
                        "Is Followed": "N/A",
                        "Rationale": metric_data.get("rationale", "")
                    })
            
            metrics_df = pd.DataFrame(metrics_data)
            
            # Create columns for visualization
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Create a radar chart for diagnostic metrics
                applicable_metrics = [metric for metric in metrics_data 
                                    if metric["Is Followed"] != "N/A"]
                
                if applicable_metrics:
                    metric_names = [metric["Metric"] for metric in applicable_metrics]
                    metric_values = [1 if metric["Is Followed"] else 0 for metric in applicable_metrics]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=metric_values,
                        theta=metric_names,
                        fill='toself',
                        name='Metrics'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )
                        ),
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No applicable diagnostic metrics found")
            
            with col2:
                # Show metrics table with rationales
                st.dataframe(
                    metrics_df.style.apply(
                        lambda x: ['background-color: #d4f1dd' if x['Is Followed'] == True else
                                ('background-color: #f1d4d4' if x['Is Followed'] == False else '') 
                                for i in range(len(x))], axis=1
                    ),
                    height=400
                )
            
            # Detailed rationale
            with st.expander("View Instruction Rationale"):
                st.write(selected_instr_data["rationale"])
        else:
            st.write("No diagnostic metrics available for this instruction.")
    else:
        st.write("No applicable instructions found for this conversation.")

def generate_if_accuracy_dashboard(data):
    """
    Generate an interactive dashboard for analyzing instruction following (IF) accuracy metrics.
    This function creates a Streamlit dashboard that displays various analyses of conversation
    data, focusing on how well instructions were followed across different segments. The dashboard
    includes multiple tabs for different types of analysis and filtering capabilities.
    Parameters:
    ----------
    data : dict
        A dictionary containing conversation analysis data with the following required keys:
        - 'all_conversations': list of conversation objects with conversation_id
        - 'all_conversation_analyses': list of analysis objects containing:
            - conversation_id
            - segment
            - conversation_if_accuracy (dict with followed_instructions, applicable_instructions)
    Returns:
    -------
    None
        Displays the Streamlit dashboard with the following components:
        - Segment filter
        - Overall accuracy metrics
        - Four analysis tabs:
            1. Segment Analysis
            2. Instruction Analysis
            3. Diagnostic Metrics Analysis
            4. Conversation Analysis
    Notes:
    -----
    Requires Streamlit and Pandas to be installed and imported.
    The function will stop execution if no conversations are found for selected segments.
    """
    
    # Create a mapping of conversation IDs to full conversation objects for reference
    conversation_map = {conv["conversation_id"]: conv for conv in data.get("all_conversations", [])}
    
    # Extract unique segments for filtering
    all_segments = sorted(list(set(conv.get("segment", "Unknown") for conv in data["all_conversation_analyses"])))
    
    st.title("IF Accuracy Analysis Dashboard")
    
    # Add segment filter as multi-select
    selected_segments = st.multiselect(
        "Filter by Segment",
        options=all_segments,
        default=all_segments,
        help="Select one or more segments to include in the analysis"
    )
    
    # Filter conversations based on selected segments
    filtered_conversations = [conv for conv in data["all_conversation_analyses"] 
                             if conv.get("segment", "Unknown") in selected_segments]
    
    # Calculate overall accuracy for filtered conversations
    if filtered_conversations:
        filtered_accuracy = sum(conv["conversation_if_accuracy"]["followed_instructions"] 
                              for conv in filtered_conversations) / sum(conv["conversation_if_accuracy"]["applicable_instructions"] 
                              for conv in filtered_conversations) * 100
    else:
        filtered_accuracy = 0
    
    show_top_level_metrics(all_segments, selected_segments, filtered_conversations, filtered_accuracy)
    
    if not filtered_conversations:
        st.warning("No conversations found for the selected segments. Please select different segments.")
        st.stop()
    
    # Create dataframe for conversations
    conversation_data = []
    for conv in filtered_conversations:
        conversation_data.append({
            "Conversation ID": conv["conversation_id"],
            "Segment": conv.get("segment", "Unknown"),
            "IF Accuracy": conv["conversation_if_accuracy"]["if_accuracy"] * 100,
            "Applicable Instructions": conv["conversation_if_accuracy"]["applicable_instructions"],
            "Followed Instructions": conv["conversation_if_accuracy"]["followed_instructions"]
        })
    
    conv_df = pd.DataFrame(conversation_data)

    tab1, tab2, tab3, tab4 = st.tabs(["Segment Analysis", "Instruction Analysis", "Diagnostic Metrics Analysis", "Conversation Analysis"])
    
    with tab1:
        show_segment_analysis(conv_df)

    with tab2:
        show_instruction_analysis(filtered_conversations, conversation_map)

    with tab3:
        show_diagnostic_metrics_analysis(filtered_conversations, conversation_map)

    with tab4:
        show_conversation_analysis(conv_df, filtered_conversations, conversation_map)