import csv
import numpy as np
import matplotlib.pyplot as plt
import ast
import seaborn as sns
import pandas as pd
import plotly.express as px
import yaml
from arcmg.config import Config
import argparse
import os
from arcmg.plot import plot_loss_histogram, make_edge, make_arrow
import network as nx
import plotly.graph_objects as go
import plotly.offline as py

def main(path_file):

    with open(os.path.join(path_file, "all_exps.yaml"), mode="rb") as yaml_reader:
        all_exp = yaml.safe_load(yaml_reader)

    train_loss_success = []
    test_loss_success = []
    train_loss_failure = []
    test_loss_failure = []

    with open(path_file + 'all_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        done_writing_header = False

        for key, value in all_exp.items():
            config = Config(value)

            with open(os.path.join(config.output_dir, "result.csv"), 'r') as file_exp:
                reader = csv.reader(file_exp)
                # if done writing the header, skip the first line of the csv file
                if done_writing_header:
                    next(reader)

                for row in reader:
                    # if done writing the header, then we skipped the first line of the csv file, so we add variables to the results that are saved
                    if done_writing_header:
                        new_row = row

                        # to do: make this not depend on the order that the information appears in case we change it
                        num_classes_learned = ast.literal_eval(row[1])
                        is_bistable = ast.literal_eval(row[2])
                        final_train = ast.literal_eval(row[3])
                        final_test = ast.literal_eval(row[4])

                        if num_classes_learned == config.num_labels and is_bistable:
                            train_loss_success.append(final_train)
                            test_loss_success.append(final_test)
                            success = True

                        else:
                            train_loss_failure.append(final_train)
                            test_loss_failure.append(final_test)
                            success = False

                        for i in [success, config.base, config.backward_exponent, config.forward_exponent, config.max_penalty, key]:
                            new_row.append(i)

                        writer.writerow(new_row)

                    else:
                        row.extend(['success', 'base', 'backward_coeff', 'forward_coeff', 'max_penalty', 'key']) 
                        writer.writerow(row)
                        done_writing_header = True

    new_all_class_sets = []
    transition_dict_apperance_number = dict()
    class_set_to_int_dict = dict()
    int_to_class_set_dict = dict()
    edge_list = []
    class_set_appearance_number = dict()

    for key, value in all_exp.items():
        config = Config(value)

        if config.analyze_train_dynamics:
            with open(os.path.join(config.output_dir, "train_dynamics.csv"), 'r') as file_dyn:
                reader = csv.reader(file_dyn)
                on_first_row = True

                for row in reader:
                    class_set = []
                    
                    for num in row:
                        class_set.append(int(num))
                    new_class_set = tuple(class_set)

                    if (not on_first_row) and new_class_set != last_class_set:
                        new_class_set_index = len(new_all_class_sets)

                        int_to_class_set_dict[new_class_set_index] = new_class_set
                        class_set_to_int_dict[new_class_set] = new_class_set_index

                        edge_list.append([class_set_to_int_dict[last_class_set], new_class_set_index])
                        try:
                            transition_dict_apperance_number[(last_class_set, new_class_set)] = transition_dict_apperance_number[(last_class_set, new_class_set)] + 1
                        except:
                            transition_dict_apperance_number[(last_class_set, new_class_set)] = 1
                        
                        try:
                            class_set_appearance_number[new_class_set] = class_set_appearance_number[new_class_set] + 1
                        except:
                            class_set_appearance_number[new_class_set] = 1

                    new_all_class_sets.append(new_class_set)
                    new_all_class_sets = list(set(new_all_class_sets))
                    
                    #if (not on_first_row) and len(new_all_class_sets) != len(last_all_class_sets):
                    #    int_to_class_set_dict[len(new_all_class_sets)] = new_class_set
                        #    class_set_to_int_dict[new_class_set] = len(new_all_class_sets)

                    if on_first_row:
                        int_to_class_set_dict[0] = new_class_set
                        class_set_to_int_dict[new_class_set] = 0
                        try:
                            class_set_appearance_number[new_class_set] = class_set_appearance_number[new_class_set] + 1
                        except:
                            class_set_appearance_number[new_class_set] = 1

                    last_class_set = new_class_set
        
                    on_first_row = False

    print('transition dict appearance number: ', transition_dict_apperance_number)
    print('all class sets: ', new_all_class_sets)
    print('int to class set dict: ', int_to_class_set_dict)
    print('class set to int dict: ', class_set_to_int_dict)
    print('edge list: ', edge_list)

    dynamics_network = nx.DiGraph()

    for class_set in class_set_appearance_number.keys():
        num_appearances = class_set_appearance_number[class_set]
        dynamics_network.add_node(class_set, size = num_appearances)

    for transition in transition_dict_apperance_number.keys():
        v1 = transition[0]
        v2 = transition[1]
        weight = transition_dict_apperance_number[transition]
        dynamics_network.add_edge(v1, v2, weight = weight)

    pos_ = nx.spring_layout(dynamics_network)

        # For each edge, make an edge_trace, append to list
    edge_trace = []
    for edge in dynamics_network.edges():

        v1 = edge[0]
        v2 = edge[1]

        x0, y0 = pos_[v1]
        x1, y1 = pos_[v2]

        text   = str(v1) + '--' + str(v2) + ': ' + str(dynamics_network.edges()[edge]['weight'])

        print('text: ', text)
        
       # trace  = make_edge([x0, x1, None], [y0, y1, None], text,
                       # 0.3*dynamics_network.edges()[edge]['weight']**1.75)

        trace  = make_arrow([x0, x1], [y0, y1],
                        dynamics_network.edges()[edge]['weight']**2.75)

        edge_trace.append(trace)

        # Make a node trace
    node_trace = go.Scatter(x         = [],
                            y         = [],
                            text      = [],
                            textposition = "top center",
                            textfont_size = 20,
                            mode      = 'markers+text',
                            hoverinfo = 'none',
                            marker    = dict(color = [],
                                            size  = [],
                                            line  = None))
    # For each node in dynamics_network, get the position and size and add to the node_trace
    for node in dynamics_network.nodes():
        x, y = pos_[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['marker']['color'] += tuple(['cornflowerblue'])
        node_trace['marker']['size'] += tuple([10*dynamics_network.nodes()[node]['size']])
        node_trace['text'] += tuple(['<b>' + str(node) + '</b>'])

    layout = go.Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
    )


    fig = go.Figure(layout = layout)

    for trace in edge_trace:
        fig.add_annotation(trace)

    fig.add_trace(node_trace)

    fig.update_layout(showlegend = False)

    fig.update_xaxes(showticklabels = False)

    fig.update_yaxes(showticklabels = False)

    fig.show()
    py.plot(fig, filename='dynamics_network.html')


    #plot_loss_histogram(train_loss_success, train_loss_failure, 'train', path_file)
    #plot_loss_histogram(test_loss_success, test_loss_failure, 'test', path_file)

    with open(path_file + 'train_loss_analysis.csv', 'w', newline = '') as file:
        writer = csv.writer(file)
        for loss_value in train_loss_success:
            writer.writerow([loss_value, 'success'])
        for loss_value in train_loss_failure:
            writer.writerow([loss_value, 'failure'])

    with open(path_file + 'test_loss_analysis.csv', 'w', newline = '') as file:
        writer = csv.writer(file)
        for loss_value in test_loss_success:
            writer.writerow([loss_value, 'success'])
        for loss_value in test_loss_failure:
            writer.writerow([loss_value, 'failure'])



    #with open(path_file + 'parameter_analysis.csv', 'w', newline='') as file:
        #writer = csv.writer(file)

        #parameters_sets_unique = set(success_parameter_sets)

        #print('parameter sets: ', parameters_sets_unique)
        #temp = 0
        #for elmt in parameters_sets_unique:
            #print('elmt: ', elmt)
            #count = success_parameter_sets.count(elmt)
            #print(str(elmt) + ': ' + "{:.2f}".format(percent))
            #temp += percent
            #writer.writerow([elmt, percent])

if __name__ == "__main__":
   path_file= "output/rampfn2024_02_22_2030"
   #"/scratch/bg545/phase-space-classification/experiments/output/rampfn2024_02_22_2030"
   
   parser = argparse.ArgumentParser()
   parser.add_argument('--path_dir',help='Directory of config files',type=str,default=path_file)
   args = parser.parse_args()
   
   main(args.path_dir)






exit()



# Warning: File under construction

num_jobs = 810

num_labels_list = []
final_loss_list = []

# to do: get this from the config file
num_labels = 3

least_loss_per_group_dict = dict()
for group in range(1, num_labels + 1):
    least_loss_per_group_dict[group] = 10

greatest_loss_per_group_dict = dict()
for group in range(1, num_labels + 1):
    greatest_loss_per_group_dict[group] = 0

with open('concatenated_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["base", "forward_exponent", "backward_exponent", "separatrix_self_penalty", "max_penalty", "test_loss", "num_classes_found"])

    job_index_final_loss_dict = dict()
    for job_index in range(0, num_jobs):
        job_index_final_loss_dict[job_index] = []

    for job_index in range(0, num_jobs):
        try:
            file_name = f'output/arctan/experiment_results/results_{job_index}.csv'
            with open(file_name) as csvfile:
                reader = csv.DictReader(csvfile)
                data = list(reader)
                for index, row in enumerate(data):

                    writer.writerow([row["base"], row["forward_exponent"], row["backward_exponent"], row["separatrix_self_penalty"], row["max_penalty"], row["test_loss"], int(row["num_classes_found"])])

        
                    loss = float(row["test_loss"])
                    print('loss is: ', loss)

                    old_final_loss_list = job_index_final_loss_dict[job_index]
                    new_final_loss_list = old_final_loss_list + [loss]
                    job_index_final_loss_dict[job_index] = new_final_loss_list

                    num_classes_found = int(row["num_classes_found"])
                    #num_labels_list.append()

                    if loss < least_loss_per_group_dict[num_classes_found]:
                        least_loss_per_group_dict[num_classes_found] = loss
                    if loss > greatest_loss_per_group_dict[num_classes_found]:
                        greatest_loss_per_group_dict[num_classes_found] = loss

                    final_loss_list.append(loss)

        except:
            pass

def get_normalized_loss(loss, num_classes_found, least_loss_per_group_dict, greatest_loss_per_group_dict):
    min = least_loss_per_group_dict[num_classes_found]
    max = greatest_loss_per_group_dict[num_classes_found]
    print('loss: ', loss)
    print('min: ', min)
    print('max: ', max)
    if max != min:
        return loss / (max - min) 
    else:
        return loss

def score_network(loss, num_classes_found, least_loss_per_group_dict, greatest_loss_per_group_dict, num_labels):
    normalized_loss = get_normalized_loss(loss, num_classes_found, least_loss_per_group_dict, greatest_loss_per_group_dict)
    num_classes_missing = num_labels - num_classes_found
    print('num classes missing: ', num_classes_missing)
    print('num classes foudn: ', num_classes_found)
    return num_classes_missing + normalized_loss


df = pd.read_csv('concatenated_results.csv')

network_score_list = []

# index is not used but currently unsure what other way to iterate through rows of the dataframe
for index, row in df.iterrows():
    test_loss = row['test_loss']
    num_classes_found = int(row['num_classes_found'])
    max_penalty = row['max_penalty']
    base = row['base']
   # print('max penalty: ', max_penalty)
   # print('base: ', base)
    network_score = score_network(test_loss, num_classes_found, least_loss_per_group_dict, greatest_loss_per_group_dict, num_labels)
    network_score_list.append(network_score)
    #print('network scores: ', network_score_list)
  #  print('----------')
#    exit()

df['network_scores'] = network_score_list
print(df)
df.to_csv('dataframe.csv')


fig = px.parallel_coordinates(df, color="network_scores",
                              dimensions=['base','forward_exponent','backward_exponent','separatrix_self_penalty','max_penalty'],
                              color_continuous_scale=px.colors.sequential.Bluered)
fig.show()

exit()

# turn final_loss_data into a dataframe
final_loss_df = pd.DataFrame(final_loss_list, columns = ["final_loss"])

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))

# Get unique values from correctness_list
unique_values = set(num_labels_list)

# Define marker styles for each unique value
#marker_styles = {value: '^' if 'Incorrect' in value else 'o' for value in unique_values}

# Generate a list of markers based on correctness_list
#marker_list = [marker_styles[value] for value in num_labels_list]

sns.swarmplot(data=final_loss_df, x="final_loss", hue=num_labels_list, palette = 'viridis', legend=True)
#ax.violinplot(final_loss_data, showmeans=False, showmedians=True, showextrema=True)

ax.set_title('Swarmplot of Final Test Loss')

#labels_N = [str(N) for N in N_range]

#ax.yaxis.grid(True)
#ax.set_xticks([y + 1 for y in range(len(final_loss_list))])
ax.set_xlabel('Final Loss')
#ax.set_ylabel('Final Loss')

filename = 'swarmplot.png'
plt.savefig(filename)

plt.show()

