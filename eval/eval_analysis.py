import os


def count_assistant_turns(messages):
    """Count the number of assistant messages in the conversation.

    For models like gpt-oss (harmony encoding) each generation round
    produces multiple assistant messages (analysis + action channels),
    so raw message counting overstates the real turn count.  Callers
    that have access to the full record should prefer count_turns_from_record.
    """
    return sum(1 for msg in messages if msg.get('role') == 'assistant')


def count_turns_from_record(record: dict) -> int:
    """Return the number of model-generation turns for a single result record.

    Uses turn_stats when available (accurate for all model families including
    gpt-oss/harmony where each round emits multiple assistant messages).
    Falls back to counting assistant messages if turn_stats is absent.
    """
    turn_stats = record.get('turn_stats') or []
    if turn_stats:
        return len(turn_stats)
    messages = record.get('full_messages') or record.get('messages') or []
    return count_assistant_turns(messages)


ALL_TOOLS = [
    'browser.search',
    'browser.open',
    'browser.find',
]


def count_tool_usage(messages):
    """
    Count browser tool usage in the conversation.
    Looks at assistant messages with tool_calls (standard OpenAI format) and
    tool-role messages that carry a name field.

    Returns:
        dict: {tool_name: count} for every tool in ALL_TOOLS
    """
    tool_counts = {t: 0 for t in ALL_TOOLS}

    for msg in messages:
        # tool-role messages with explicit name field
        if msg.get('role') == 'tool' and 'name' in msg:
            tool_name = msg['name']
            for key in ALL_TOOLS:
                if key in tool_name:
                    tool_counts[key] += 1
                    break
            else:
                if 'channel' in msg and msg['channel'] in tool_counts:
                    tool_counts[msg['channel']] += 1

        # assistant messages with tool_calls (primary source)
        if msg.get('role') == 'assistant' and msg.get('tool_calls'):
            for tool_call in msg['tool_calls']:
                tool_name = tool_call.get('function', {}).get('name', '')
                for key in ALL_TOOLS:
                    if key in tool_name:
                        tool_counts[key] += 1
                        break

    return tool_counts


def collect_turn_data(correct_items, incorrect_items, qid_to_data):
    """
    Collect turn counts for correct and incorrect answers.

    Args:
        correct_items: List of correctly answered items
        incorrect_items: List of incorrectly answered items
        qid_to_data: Dict mapping qid to original data with messages

    Returns:
        tuple: (correct_turns, incorrect_turns) lists
    """
    correct_turns = []
    incorrect_turns = []

    for item in correct_items:
        qid = item['qid']
        if qid in qid_to_data:
            turns = count_turns_from_record(qid_to_data[qid])
            correct_turns.append(turns)

    for item in incorrect_items:
        qid = item['qid']
        if qid in qid_to_data:
            turns = count_turns_from_record(qid_to_data[qid])
            incorrect_turns.append(turns)

    return correct_turns, incorrect_turns


def collect_tool_usage_data(correct_items, incorrect_items, qid_to_data):
    """
    Collect tool usage counts for correct and incorrect answers.

    Args:
        correct_items: List of correctly answered items
        incorrect_items: List of incorrectly answered items
        qid_to_data: Dict mapping qid to original data with messages

    Returns:
        tuple: (correct_tool_usage, incorrect_tool_usage) lists of dicts
    """
    correct_tool_usage = []
    incorrect_tool_usage = []

    for item in correct_items:
        qid = item['qid']
        if qid in qid_to_data:
            tool_counts = count_tool_usage((qid_to_data[qid].get('full_messages') or qid_to_data[qid].get('messages') or []))
            correct_tool_usage.append(tool_counts)

    for item in incorrect_items:
        qid = item['qid']
        if qid in qid_to_data:
            tool_counts = count_tool_usage((qid_to_data[qid].get('full_messages') or qid_to_data[qid].get('messages') or []))
            incorrect_tool_usage.append(tool_counts)

    return correct_tool_usage, incorrect_tool_usage


def print_turn_statistics(correct_turns, incorrect_turns):
    """Print turn distribution statistics"""
    print("\n" + "="*60)
    print("Turn Distribution Analysis")
    print("="*60)

    if correct_turns:
        print(f"\nCorrect Answers (n={len(correct_turns)}):")
        print(f"  Mean turns: {sum(correct_turns)/len(correct_turns):.2f}")
        print(f"  Median turns: {sorted(correct_turns)[len(correct_turns)//2]:.2f}")
        print(f"  Min turns: {min(correct_turns)}")
        print(f"  Max turns: {max(correct_turns)}")

    if incorrect_turns:
        print(f"\nIncorrect Answers (n={len(incorrect_turns)}):")
        print(f"  Mean turns: {sum(incorrect_turns)/len(incorrect_turns):.2f}")
        print(f"  Median turns: {sorted(incorrect_turns)[len(incorrect_turns)//2]:.2f}")
        print(f"  Min turns: {min(incorrect_turns)}")
        print(f"  Max turns: {max(incorrect_turns)}")

    print("="*60)


def create_turn_distribution_plots(correct_turns, incorrect_turns, output_dir):
    """
    Create visualization plots for turn distribution analysis.

    Args:
        correct_turns: List of turn counts for correct answers
        incorrect_turns: List of turn counts for incorrect answers
        output_dir: Directory to save the plots
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import numpy as np

        # Create output directory for plots
        plot_dir = output_dir.rstrip('/') + '/plots'
        os.makedirs(plot_dir, exist_ok=True)

        # Figure 1: Histograms side by side
        _create_side_by_side_histograms(correct_turns, incorrect_turns, plot_dir)

        # Figure 2: Box plot comparison
        # _create_boxplot(correct_turns, incorrect_turns, plot_dir)

        # # Figure 3: Cumulative Distribution Function (CDF)
        # _create_cdf_plot(correct_turns, incorrect_turns, plot_dir)

        # Figure 4: Overlaid histogram with transparency
        # _create_overlay_histogram(correct_turns, incorrect_turns, plot_dir)

        print(f"\nAll plots saved to directory: {plot_dir}")

    except ImportError:
        print("\nNote: matplotlib not installed. Install with 'pip install matplotlib' to generate plots.")
    except Exception as e:
        print(f"\nWarning: Could not generate plots: {e}")


def _create_side_by_side_histograms(correct_turns, incorrect_turns, plot_dir):
    """Create side-by-side histograms for correct and incorrect answers"""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if correct_turns:
        axes[0].hist(correct_turns, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[0].axvline(np.mean(correct_turns), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(correct_turns):.1f}')
        axes[0].axvline(np.median(correct_turns), color='blue', linestyle='--',
                       linewidth=2, label=f'Median: {np.median(correct_turns):.1f}')
        axes[0].set_xlabel('Number of Turns')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Correct Answers (n={len(correct_turns)})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    if incorrect_turns:
        axes[1].hist(incorrect_turns, bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[1].axvline(np.mean(incorrect_turns), color='darkred', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(incorrect_turns):.1f}')
        axes[1].axvline(np.median(incorrect_turns), color='blue', linestyle='--',
                       linewidth=2, label=f'Median: {np.median(incorrect_turns):.1f}')
        axes[1].set_xlabel('Number of Turns')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Incorrect Answers (n={len(incorrect_turns)})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    hist_path = os.path.join(plot_dir, 'turn_distribution_histograms.png')
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nHistogram saved to: {hist_path}")


def _create_boxplot(correct_turns, incorrect_turns, plot_dir):
    """Create box plot comparison"""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 6))

    data_to_plot = []
    labels = []
    if correct_turns:
        data_to_plot.append(correct_turns)
        labels.append(f'Correct\n(n={len(correct_turns)})')
    if incorrect_turns:
        data_to_plot.append(incorrect_turns)
        labels.append(f'Incorrect\n(n={len(incorrect_turns)})')

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                   showmeans=True, meanline=True)

    colors = ['lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
        patch.set_facecolor(color)

    ax.set_ylabel('Number of Turns')
    ax.set_title('Turn Distribution Comparison: Correct vs Incorrect Answers')
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistics text
    stats_text = []
    if correct_turns:
        stats_text.append(f"Correct: Mean={np.mean(correct_turns):.1f}, Median={np.median(correct_turns):.1f}")
    if incorrect_turns:
        stats_text.append(f"Incorrect: Mean={np.mean(incorrect_turns):.1f}, Median={np.median(incorrect_turns):.1f}")

    ax.text(0.02, 0.98, '\n'.join(stats_text), transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    boxplot_path = os.path.join(plot_dir, 'turn_distribution_boxplot.png')
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Boxplot saved to: {boxplot_path}")


def _create_cdf_plot(correct_turns, incorrect_turns, plot_dir):
    """Create Cumulative Distribution Function plot"""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 6))

    if correct_turns:
        sorted_correct = np.sort(correct_turns)
        cdf_correct = np.arange(1, len(sorted_correct) + 1) / len(sorted_correct)
        ax.plot(sorted_correct, cdf_correct, label=f'Correct (n={len(correct_turns)})',
               linewidth=2, color='green')

    if incorrect_turns:
        sorted_incorrect = np.sort(incorrect_turns)
        cdf_incorrect = np.arange(1, len(sorted_incorrect) + 1) / len(sorted_incorrect)
        ax.plot(sorted_incorrect, cdf_incorrect, label=f'Incorrect (n={len(incorrect_turns)})',
               linewidth=2, color='red')

    ax.set_xlabel('Number of Turns')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Function: Turn Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    cdf_path = os.path.join(plot_dir, 'turn_distribution_cdf.png')
    plt.savefig(cdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"CDF plot saved to: {cdf_path}")


def _create_overlay_histogram(correct_turns, incorrect_turns, plot_dir):
    """Create overlaid histogram with transparency"""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 6))

    if correct_turns and incorrect_turns:
        # Determine common bin range
        all_turns = correct_turns + incorrect_turns
        bins = np.linspace(min(all_turns), max(all_turns), 40)

        ax.hist(correct_turns, bins=bins, alpha=0.5, color='green',
               label=f'Correct (n={len(correct_turns)})', edgecolor='black')
        ax.hist(incorrect_turns, bins=bins, alpha=0.5, color='red',
               label=f'Incorrect (n={len(incorrect_turns)})', edgecolor='black')

        ax.axvline(np.mean(correct_turns), color='darkgreen', linestyle='--',
                  linewidth=2, label=f'Correct Mean: {np.mean(correct_turns):.1f}')
        ax.axvline(np.mean(incorrect_turns), color='darkred', linestyle='--',
                  linewidth=2, label=f'Incorrect Mean: {np.mean(incorrect_turns):.1f}')

    ax.set_xlabel('Number of Turns')
    ax.set_ylabel('Frequency')
    ax.set_title('Turn Distribution: Correct vs Incorrect Answers (Overlaid)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    overlay_path = os.path.join(plot_dir, 'turn_distribution_overlay.png')
    plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Overlay histogram saved to: {overlay_path}")


def create_tool_usage_plots(correct_tool_usage, incorrect_tool_usage, output_dir):
    """
    Create bar plots for tool usage comparison.

    Args:
        correct_tool_usage: List of tool count dicts for correct answers
        incorrect_tool_usage: List of tool count dicts for incorrect answers
        output_dir: Directory to save the plots
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import numpy as np

        plot_dir = output_dir.rstrip('/') + '/plots'
        os.makedirs(plot_dir, exist_ok=True)

        def aggregate_tool_stats(tool_list):
            if not tool_list:
                return None
            total = {t: sum(item.get(t, 0) for item in tool_list) for t in ALL_TOOLS}
            n = len(tool_list)
            avg = {t: total[t] / n for t in ALL_TOOLS}
            return {'total': total, 'average': avg, 'count': n}

        fig, ax = plt.subplots(figsize=(13, 6))

        tools = ALL_TOOLS
        tool_labels = ['Search', 'Open', 'Find']

        if correct_tool_usage and incorrect_tool_usage:
            correct_stats = aggregate_tool_stats(correct_tool_usage)
            incorrect_stats = aggregate_tool_stats(incorrect_tool_usage)

            x = np.arange(len(tools))
            width = 0.35

            correct_avgs = [correct_stats['average'][t] for t in tools]
            incorrect_avgs = [incorrect_stats['average'][t] for t in tools]
            total_calls = [
                correct_stats['total'][t] + incorrect_stats['total'][t]
                for t in tools
            ]
            xtick_labels = [
                f'{label}\nTotal: {total:,}'
                for label, total in zip(tool_labels, total_calls)
            ]

            bars1 = ax.bar(x - width/2, correct_avgs, width,
                           label=f'Correct (n={correct_stats["count"]})',
                           color='green', alpha=0.7, edgecolor='black')
            bars2 = ax.bar(x + width/2, incorrect_avgs, width,
                           label=f'Incorrect (n={incorrect_stats["count"]})',
                           color='red', alpha=0.7, edgecolor='black')

            ax.set_xlabel('Tool Type', fontsize=12)
            ax.set_ylabel('Average Usage per Question', fontsize=12)
            ax.set_title('Average Tool Usage per Question: Correct vs Incorrect',
                         fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(xtick_labels)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        tool_usage_path = os.path.join(plot_dir, 'tool_usage_comparison.png')
        plt.savefig(tool_usage_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nTool usage plot saved to: {tool_usage_path}")

    except ImportError:
        print("\nNote: matplotlib not installed. Install with 'pip install matplotlib' to generate plots.")
    except Exception as e:
        print(f"\nWarning: Could not generate tool usage plots: {e}")


_TOKENIZER_CACHE = {}

def _get_tokenizer(model_name):
    if model_name not in _TOKENIZER_CACHE:
        try:
            from transformers import AutoTokenizer
            _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            _TOKENIZER_CACHE[model_name] = None
    return _TOKENIZER_CACHE[model_name]


def compute_context_length(messages, model_name='openai/gpt-oss-120b'):
    """Compute total token count across all messages using the model's own tokenizer."""
    tokenizer = _get_tokenizer(model_name)

    def _count(text):
        if tokenizer is not None:
            return len(tokenizer.encode(text, add_special_tokens=False))
        return len(text) // 4  # fallback: ~4 chars per token

    total = 0
    for msg in messages:
        content = msg.get('content', '')
        if isinstance(content, str):
            total += _count(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += _count(str(part.get('text', '') or part.get('content', '')))
        tool_calls = msg.get('tool_calls', [])
        if tool_calls:
            for tc in tool_calls:
                total += _count(str(tc.get('function', {}).get('arguments', '')))
    return total


def create_turns_accuracy_plot(parsed_output, qid_to_data, output_dir):
    """
    Create a turns interval vs accuracy plot using bucketed turn counts.

    Args:
        parsed_output: List of judged items with 'correct' field
        qid_to_data: Dict mapping qid to original data with messages
        output_dir: Directory to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import numpy as np

        plot_dir = output_dir.rstrip('/') + '/plots'
        os.makedirs(plot_dir, exist_ok=True)

        # Collect (turns, correct) pairs
        turns_list, corrects = [], []
        for item in parsed_output:
            qid = item['qid']
            if qid not in qid_to_data:
                continue
            msgs = (qid_to_data[qid].get('full_messages') or qid_to_data[qid].get('messages') or [])
            turns = count_assistant_turns(msgs)
            turns_list.append(turns)
            corrects.append(1 if item.get('correct') else 0)

        if not turns_list:
            return

        turns_arr = np.array(turns_list)
        corrects_arr = np.array(corrects)

        # Bucket into percentile-based bins for even sample distribution
        n_bins = 10
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(turns_arr, percentiles)
        bin_edges = np.unique(bin_edges)

        bin_accuracies, bin_centers, bin_counts, bin_labels = [], [], [], []
        for i in range(len(bin_edges) - 1):
            mask = (turns_arr >= bin_edges[i]) & (turns_arr < bin_edges[i + 1])
            if i == len(bin_edges) - 2:
                mask = (turns_arr >= bin_edges[i]) & (turns_arr <= bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            bin_accuracies.append(corrects_arr[mask].mean() * 100)
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_counts.append(mask.sum())
            lo, hi = int(bin_edges[i]), int(bin_edges[i + 1])
            bin_labels.append(f'{lo}–{hi}')

        fig, ax1 = plt.subplots(figsize=(12, 6))

        overall_accuracy = corrects_arr.mean() * 100

        color_acc = '#2196F3'
        ax1.plot(range(len(bin_centers)), bin_accuracies, 'o-', color=color_acc,
                 linewidth=2, markersize=8, label='Accuracy (%)')
        ax1.axhline(overall_accuracy, color='orange', linestyle='--', linewidth=1.5,
                    label=f'Overall accuracy ({overall_accuracy:.1f}%)')
        ax1.set_xlabel('Turns Bucket (low → high)', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12, color=color_acc)
        ax1.tick_params(axis='y', labelcolor=color_acc)
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)

        # Add sample count as bar chart on secondary axis
        ax2 = ax1.twinx()
        ax2.bar(range(len(bin_centers)), bin_counts, alpha=0.2, color='gray', label='Sample count')
        ax2.set_ylabel('Sample Count', fontsize=11, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

        ax1.set_xticks(range(len(bin_centers)))
        ax1.set_xticklabels(bin_labels, rotation=30, ha='right', fontsize=9)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.title('Turns vs Accuracy', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, 'turns_accuracy.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nTurns–accuracy plot saved to: {plot_path}")

    except ImportError:
        print("\nNote: matplotlib not installed. Install with 'pip install matplotlib' to generate plots.")
    except Exception as e:
        print(f"\nWarning: Could not generate turns–accuracy plot: {e}")


def create_context_length_accuracy_plot(parsed_output, qid_to_data, output_dir, model_name='openai/gpt-oss-120b'):
    """
    Create a context length vs accuracy plot using bucketed context lengths.

    Args:
        parsed_output: List of judged items with 'correct' field
        qid_to_data: Dict mapping qid to original data with messages
        output_dir: Directory to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import numpy as np

        plot_dir = output_dir.rstrip('/') + '/plots'
        os.makedirs(plot_dir, exist_ok=True)

        # Collect (context_length, correct, latency) triples
        lengths, corrects, latencies = [], [], []
        for item in parsed_output:
            qid = item['qid']
            if qid not in qid_to_data:
                continue
            record = qid_to_data[qid]
            ctx_len = compute_context_length((record.get('final_messages') or record.get('messages') or []), model_name=model_name)
            lengths.append(ctx_len)
            corrects.append(1 if item.get('correct') else 0)
            latency = record.get('latency_s')
            if isinstance(latency, (int, float)):
                latencies.append(float(latency))

        if not lengths:
            return

        lengths = np.array(lengths)
        corrects = np.array(corrects)

        # Equal-width bins so sample counts reflect the real context-length distribution.
        # Use ~10 bins spanning [min, max], snapped to round numbers for readability.
        n_bins = 10
        bin_edges = np.linspace(lengths.min(), lengths.max(), n_bins + 1)
        bin_edges = np.unique(bin_edges)

        bin_accuracies, bin_centers, bin_counts = [], [], []
        for i in range(len(bin_edges) - 1):
            mask = (lengths >= bin_edges[i]) & (lengths < bin_edges[i + 1])
            if i == len(bin_edges) - 2:
                mask = (lengths >= bin_edges[i]) & (lengths <= bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            bin_accuracies.append(corrects[mask].mean() * 100)
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_counts.append(mask.sum())

        fig, ax1 = plt.subplots(figsize=(12, 6))

        overall_accuracy = corrects.mean() * 100

        color_acc = '#2196F3'
        ax1.plot(range(len(bin_centers)), bin_accuracies, 'o-', color=color_acc,
                 linewidth=2, markersize=8, label='Accuracy (%)')
        ax1.axhline(overall_accuracy, color='orange', linestyle='--', linewidth=1.5,
                    label=f'Overall accuracy ({overall_accuracy:.1f}%)')
        ax1.set_xlabel('Context Length Bucket in Tokens (low → high)', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12, color=color_acc)
        ax1.tick_params(axis='y', labelcolor=color_acc)
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)

        # Add sample count as bar chart on secondary axis
        ax2 = ax1.twinx()
        ax2.bar(range(len(bin_centers)), bin_counts, alpha=0.2, color='gray', label='Sample count')
        ax2.set_ylabel('Sample Count', fontsize=11, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

        # X-tick labels: show context length range in K tokens
        xtick_labels = [f'{bin_edges[i]/1000:.0f}K–{bin_edges[i+1]/1000:.0f}K'
                        for i in range(len(bin_edges) - 1)
                        if any((lengths >= bin_edges[i]) & (lengths <= bin_edges[i + 1]))]
        ax1.set_xticks(range(len(bin_centers)))
        ax1.set_xticklabels(xtick_labels, rotation=30, ha='right', fontsize=9)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if latencies:
            import matplotlib.lines as mlines
            avg_latency = float(np.mean(latencies))
            lines1.append(mlines.Line2D([], [], color='none'))
            labels1.append(f'Latency/q ({avg_latency:.1f}s)')
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.title('Context Length vs Accuracy', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, 'context_length_accuracy.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nContext length–accuracy plot saved to: {plot_path}")

    except ImportError:
        print("\nNote: matplotlib not installed. Install with 'pip install matplotlib' to generate plots.")
    except Exception as e:
        print(f"\nWarning: Could not generate context length–accuracy plot: {e}")
