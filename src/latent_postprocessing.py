import matplotlib.pyplot as plt  
import numpy as np
import os
import sys
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from multiprocessing import Pool
import matplotlib.cm as cm
import tensorflow as tf
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.dataloader import DataLoader
from src.config_vae import get_config
from scipy.optimize import differential_evolution
import threading
from tkinter import messagebox

class Postprocessing_Base:
    def __init__(self):
        pass

class PostprocessingVisualizer:
    def __init__(self, root, data: DataLoader):
        
        self.root = root
        self.data = data

        self.config = get_config(self.data.result_folder + '/conf.json')
        filtered = self.config["filter"]

        self.latent_space = data.model["latent_space"]
        self.gain = data.dataset['values']
        self.time = data.dataset['time']
        self.encoder = data.model["encoder"]
        self.decoder = data.model["decoder"]
        self.rna_gain = data.model["latent_gain"]
        
        self.vae_norm = self.data.vae_norm 
        self.gain_norm = self.data.gain_norm 

        key = list(filtered.keys())[0]
        gain_val = np.array(self.gain[key])
        mask = gain_val >= filtered[key]

        for key in self.gain.keys():
            self.gain[key] = np.array(self.gain[key])[mask]


        self.axis_x, self.axis_y = 0, 1  # Default dimensions to plot
        self._N = 50
        self.index = 0
        
        self._area = []
        self.x_max = None

        self.dim = self.latent_space.shape[1]
        
        # Setup main Tkinter window and frames
        self.root = root
        self.root.title("Interactive Visualization")
        
        # Frame for controls
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Axis selection dropdowns
        tk.Label(control_frame, text="X-axis:").pack(side=tk.TOP)
        self.x_axis_var = tk.IntVar(value=self.axis_x)
        x_axis_spinbox = ttk.Spinbox(control_frame, from_=0, to=self.dim-1, textvariable=self.x_axis_var, command=self.update_axes, width=3)
        x_axis_spinbox.pack(side=tk.TOP)

        tk.Label(control_frame, text="Y-axis:").pack(side=tk.TOP)
        self.y_axis_var = tk.IntVar(value=self.axis_y)
        y_axis_spinbox = ttk.Spinbox(control_frame, from_=0, to=self.dim-1, textvariable=self.y_axis_var, command=self.update_axes, width=3)
        y_axis_spinbox.pack(side=tk.TOP)
        
        # Integer entry label and widget
        tk.Label(control_frame, text="PCA dim :").pack(side=tk.TOP)
        self._pca_dim = tk.IntVar(value=2)  # Default integer value
        int_entry = tk.Entry(control_frame, textvariable=self._pca_dim, width=3)
        int_entry.pack(side=tk.TOP)
       
        tk.Label(control_frame, text="Enable PCA :").pack(side=tk.TOP)
        self.enablePCA = tk.BooleanVar(value=False) # Default to PCA enabled
        enablePCA_checkbutton = tk.Checkbutton(control_frame, text="Enable", variable=self.enablePCA, command=self.update_axes)
        enablePCA_checkbutton.pack(side=tk.TOP)
        
        options = list(self.gain.keys())
        tk.Label(control_frame, text="Select value").pack(side=tk.TOP)
        self.gain_entry = tk.StringVar(value="gain") # Default to heatmap based on gain
        self.gain_entry.trace_add("write", self.plot_main)
        gain_entry_menu = tk.OptionMenu(control_frame, self.gain_entry, *options)
        gain_entry_menu.pack(side=tk.TOP)

        # Integer entry label and widget
        tk.Label(control_frame, text="N:").pack(side=tk.TOP)
        self._N = tk.IntVar(value=50)  # Default integer value
        int_entry = tk.Entry(control_frame, textvariable=self._N, width=3)
        int_entry.pack(side=tk.TOP)
        
        self.quit_button = tk.Button(control_frame, text="mapping", command=self.plot_mapping)
        self.quit_button.pack(pady=10)       
        
        self.quit_button = tk.Button(control_frame, text="optimize", command=self.find_best)
        self.quit_button.pack(pady=10)   
        
        # Frame for matplotlib figure
        fig_frame = tk.Frame(root)
        fig_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Create main matplotlib figure and canvas
        self.fig_main, self.ax_main = plt.subplots()
        self.canvas_main = FigureCanvasTkAgg(self.fig_main, master=fig_frame)
        self.canvas_main.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Bind events
        self.canvas_main.mpl_connect("button_press_event", self.on_click)
        self.canvas_main.mpl_connect("button_press_event", self.on_press)
        self.canvas_main.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas_main.mpl_connect("button_release_event", self.on_release)
        
        self.drawing = False  # Track if the mouse is pressed
        
        # Add a button to open the input window
        # self.prepro = ttk.Button(root, text="Preprocess data", command=self.preprocessing)
        # self.prepro.pack(pady=5)
        # self.prepro_window = None
        # self.preprocessing_map = {key: lambda x: x for key in self.gain}
        
        # Create a secondary window for the detail plot
        self.detail_window = tk.Toplevel(root)
        self.detail_window.title("Detail Plot")
        self.fig_detail, self.ax_detail = plt.subplots()
        self.canvas_detail = FigureCanvasTkAgg(self.fig_detail, master=self.detail_window)
        self.canvas_detail.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.save_latent_space = tk.Button(control_frame, text="slices", command=self.show_latent_space)
        self.save_latent_space.pack(pady=5)
        
        self.quit_button = tk.Button(self.root, text="Quit", command=self.quit_app)
        self.quit_button.pack(pady=10)  
             
        # Handle window close (X button)
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
        
        # Initial plot
        self.plot_main()

    def preprocessing(self):
        if self.prepro_window is not None: return
        
        self.prepro_window = tk.Toplevel(self.root)
        self.prepro_window.title("Preprocessing")
        self.prepro_window.geometry('200x300')
        self.prepro_window.protocol("WM_DELETE_WINDOW", self.close_input_window)

        self.entries = {}
        for key in self.gain:
            label = ttk.Label(self.prepro_window, text=f"Input for {key}:") 
            label.pack(anchor="w", padx=10, pady=5)
            entry = ttk.Entry(self.prepro_window)
            entry.pack(fill=tk.X, padx=10, pady=2)
            self.entries[key] = entry

        # Add an "Update Plots" button
        update_button = ttk.Button(self.prepro_window, text="Update", command=self.update_preprocessing)
        update_button.pack(pady=10)
    
    def update_preprocessing(self):
        for key in self.gain:
            if self.entries[key].get() != '':
                self.preprocessing_map[key] = eval('lambda x:' + self.entries[key].get(), {"np": np}) 
        
    def close_input_window(self):
        self.prepro_window.destroy()
        self.prepro_window = None

    def quit_app(self):
        print("Application is closing...")  # For debugging/cleanup purposes
        self.root.quit()
        self.root.destroy() 

    def update_axes(self, *args, **kwargs):
        self.axis_x = self.x_axis_var.get()
        self.axis_y = self.y_axis_var.get()
        
        dim = self._get_dim()
        
        if self.axis_x >= dim:
            self.axis_x = dim - 1
        if self.axis_y >= dim:
            self.axis_y = dim - 1
        if self.axis_x < 0:
            self.axis_x = 0
        if self.axis_y < 0:
            self.axis_y = 0
        
        self.plot_main()

    def plot_main(self, *args, **kwargs):
        self.ax_main.clear()
        
        gg = self.gain[self.gain_entry.get()]
        
        self._pca = PCA(n_components=self._pca_dim.get())

        if self.enablePCA.get():
            latent_space = self._pca.fit_transform(self.latent_space)
            explained_var = self._pca.explained_variance_ratio_
            sc = self.ax_main.scatter(latent_space[:, self.axis_x], latent_space[:, self.axis_y], c = gg)
            self.ax_main.set_xlabel(f"Dimension {self.axis_x} ({explained_var[self.axis_x]:.2f})")
            self.ax_main.set_ylabel(f"Dimension {self.axis_y} ({explained_var[self.axis_y]:.2f})")

        else:
            sc = self.ax_main.scatter(self.latent_space[:, self.axis_x], self.latent_space[:, self.axis_y], c = gg)
            self.ax_main.set_xlabel(f"Dimension {self.axis_x}")
            self.ax_main.set_ylabel(f"Dimension {self.axis_y}")
            
        if hasattr(self, 'scatter_cb'):
            self.scatter_cb.update_normal(sc)
        else:
            self.scatter_cb = self.fig_main.colorbar(sc, ax=self.ax_main)

        self.canvas_main.draw()

    def plot_detail(self, coord):
        # Create a detailed plot in the second figure based on clicked coordinates
        self.ax_detail.clear()
        
        gain_entry = self.gain_entry.get()
        
        if self.enablePCA.get():
            dim = self._pca_dim.get()
        else:
            dim = self.dim
        
        axis = []
        
        for i in list(set(range(dim)) - {self.axis_x, self.axis_y}):
            axis.append(i)
        
        latent_point = np.zeros((1,dim))
        latent_point[0,self.axis_x] = coord[0]
        latent_point[0,self.axis_y] = coord[1]
        if self.x_max is not None and len(axis)>0:
            for a in axis:
                latent_point[0,a] = self.x_max[a]
        
        if self.enablePCA.get():
            latent_point = self._pca.inverse_transform(latent_point[0]).reshape(1,self.dim)
        else:
            dim = self.dim

        laser = self.decoder.predict(latent_point,verbose=0)[0]
        gain_val = self.rna_gain[gain_entry].predict(latent_point,verbose=0)
        gain_val = self.gain_norm[gain_entry] * (gain_val)
    
        self.ax_detail.plot(self.time, laser * self.vae_norm, label=f" {gain_entry}={gain_val}")
        self.ax_detail.set_title("Profiles")
        self.ax_detail.legend()
        self.canvas_detail.draw()
        
    def save_mapping(self):
        laser_decoded = self.decoder.predict(self.decoding_dataset,verbose=0)
        value_entry = self.gain_entry.get()
        
        folder = os.path.join(self.data.result_folder,'/decoding')
        if not os.path.exists(folder):
            os.mkdir(folder)

        for i in tqdm(range(laser_decoded.shape[0])):
            np.savetxt(folder+f"{value_entry}_{i}.dat",
                list(zip(self.time,np.abs(laser_decoded[i]))))

    def _update_plot(self):
        self.ax_mapping.clear()
        im = self.ax_mapping.pcolormesh(self._mesh[0],self._mesh[1], self._value, cmap='viridis')
        self.canvas_mapping.draw()

    def _optimize(self):
        
        if self.enablePCA.get():
            dim = self._pca_dim.get()
        else:
            dim = self.dim
        
        bounds = [(-3, 3)] * dim 
        random_samples = np.array([np.random.uniform(low, high, size=50000) for low, high in bounds]).T
        print('Random samples:', random_samples.shape)
        print('Random samples:', random_samples[0])
        
        if self.enablePCA.get():
            pca_random_samples = (random_samples)
            random_samples = self._pca.inverse_transform(pca_random_samples)
        
        predictions = np.exp(self.rna_gain[self.gain_entry.get()].predict(random_samples, verbose=0))
        
        # Find the maximum
        max_index = np.argmax(predictions)
        if self.enablePCA.get():
            best_x = pca_random_samples[max_index]
        else:
            best_x = random_samples[max_index]
            
        max_value = predictions[max_index]

        print("Best x:", best_x)
        print("Max value:", max_value)
        
        self.update_slider_values(best_x)


        messagebox.showinfo("Optimization done", f"f(x_max) = {max_value}")
        
        self.x_max = best_x

    def find_best(self):
        threading.Thread(target=self._optimize).start()
        
    def _get_min_max(self):
        xmin = min(np.array(self._area)[:,0])
        xmax = max(np.array(self._area)[:,0])
        ymin = min(np.array(self._area)[:,1])
        ymax = max(np.array(self._area)[:,1])
        
        return(xmin,xmax,ymin,ymax)
    
    def _plot_mapping(self,x_axis,y_axis):
        
        xmin,xmax,ymin,ymax = self._get_min_max()
                
        n = self._N.get()
        
        x = np.linspace(xmin,xmax,n)
        y = np.linspace(ymin,ymax,n)
        mesh = np.meshgrid(x,y)
        grid = np.vstack([m.flatten() for m in mesh]).T
        axis = []

        if self.enablePCA.get() and self._pca_dim.get() == 2:
            unfit = self._pca.inverse_transform(grid)
        elif self.enablePCA.get() and self._pca_dim.get() > 2:
            for i in list(set(range(self._pca_dim.get())) - {x_axis, y_axis}):
                axis.append(i)
            unfit = np.zeros((grid.shape[0],self._pca_dim.get() ))                
            unfit[:,x_axis] = grid[:,0]
            unfit[:,y_axis] = grid[:,1]
            if self.x_max is not None:
                for a in axis:
                    unfit[:,a] = self.x_max[a]
            unfit = self._pca.inverse_transform(unfit)
        else:
            for i in list(set(range(self.dim)) - {x_axis, y_axis}):
                axis.append(i)
            unfit = np.zeros((grid.shape[0],self.dim ))
            unfit[:,x_axis] = grid[:,0]
            unfit[:,y_axis] = grid[:,1]
            if self.x_max is not None:
                for a in axis:
                    unfit[:,a] = self.x_max[a]
        
        return mesh,unfit
    
    def _get_dim(self):
        if self.enablePCA.get():
            dim = self._pca_dim.get()
        else:
            dim = self.dim
        return dim

    def plot_mapping(self):
        
        self.mapping_window = tk.Toplevel(self.root)
        self.mapping_window.title("Mapping")
        
        # Create the button
        button = tk.Frame(self.mapping_window)
        button.pack(pady=10)  # Use pack() to add the button to the window with some padding
        
        tk.Button(button, text="Save", command=self.save_mapping).grid(row=0, column=0)

        self.fig_mapping, self.ax_mapping = plt.subplots()
        self.canvas_mapping = FigureCanvasTkAgg(self.fig_mapping, master=self.mapping_window)
        self.canvas_mapping.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_mapping.mpl_connect("button_press_event", self.on_click)
        self.canvas_mapping.mpl_connect("button_press_event", self.on_press)
        self.canvas_mapping.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas_mapping.mpl_connect("button_release_event", self.on_release)
        
        n = self._N.get()
        mesh, unfit = self._plot_mapping(self.axis_x,self.axis_y)            
        
        value_entry = self.gain_entry.get()
        dataset = tf.data.Dataset.from_tensor_slices(unfit).batch(256)
        self.decoding_dataset = dataset
        value = self.rna_gain[value_entry].predict(dataset, verbose=0).reshape((n,n))
        value = self.gain_norm[value_entry] * (value)
        im = self.ax_mapping.pcolormesh(mesh[0],mesh[1], value, cmap='viridis')
        self.ax_mapping.set_aspect('equal')
        
        if hasattr(self, 'mapping_cb'):
            self.mapping_cb.remove()  
        self.mapping_cb = self.fig_mapping.colorbar(im, ax=self.ax_mapping)
 
        self.ax_mapping.set_title(value_entry)
        self.canvas_mapping.draw()
    
    def update_slider_values(self, new_values):
        """
        Update the slider values programmatically.
        """
        for i, value in enumerate(new_values):
            # Temporarily disable the callback to avoid infinite loops
            self.sliders[i].config(command=lambda value, idx=i: None)  # Disable callback
            self.slider_vars[i].set(float(value))  # Update the slider value
            self.sliders[i].config(command=lambda value, idx=i: self._on_slider_change(idx, value))  # Re-enable callback

        # Update the plots
        self._update_latent_space_plots()
        
    def show_latent_space(self):
        dim = self._get_dim()  # Get the dimensionality of the latent space
        n = self._N.get()  # Get the resolution for the mapping

        # Create a new window for the latent space visualization
        mapping_window_all = tk.Toplevel(self.root)
        mapping_window_all.title("Latent Space Slices with Sliders")

        # Create a frame for the sliders
        slider_frame = tk.Frame(mapping_window_all)
        slider_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Create sliders for each dimension
        self.slider_vars = []  # Store slider variables
        self.sliders = []  # Store slider widgets
        xmin, xmax, ymin, ymax = self._get_min_max()
        for i in range(dim):
            label = tk.Label(slider_frame, text=f"Dim {i}:")
            label.pack(pady=5)

            # Create a slider for the current dimension
            slider_var = tk.DoubleVar(value=0.0)  # Default value for the slider
            slider = tk.Scale(
                slider_frame,
                from_=min(xmin, ymin),  # Minimum value for the latent space dimension
                to=max(xmax, ymax),      # Maximum value for the latent space dimension
                resolution=0.1,  # Step size for the slider
                orient=tk.HORIZONTAL,
                variable=slider_var,
                length=200,
                command=lambda value, idx=i: self._on_slider_change(idx, value), 
            )
            slider.pack(pady=5)

            # Store the slider and its variable
            self.slider_vars.append(slider_var)
            self.sliders.append(slider)

        # Add a button to update the plots
        update_button = tk.Button(slider_frame, text="Update Plots", command=self._update_latent_space_plots)
        update_button.pack(pady=10)

        # Create a frame for the plots
        plot_frame = tk.Frame(mapping_window_all)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create the matplotlib figure and canvas
        self.fig_mapping_all, self.ax_mapping_all = plt.subplots(dim, dim, figsize=(12, 12))
        self.canvas_mapping_all = FigureCanvasTkAgg(self.fig_mapping_all, master=plot_frame)
        self.canvas_mapping_all.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Draw the initial plots
        self._update_latent_space_plots()
        
    def _on_slider_change(self, index, value):
        """
        Callback function triggered when a slider is moved.
        """
        # Update the slider variable
        self.slider_vars[index].set(float(value))

  

    def _update_latent_space_plots(self):
        """
        Update the pairwise latent space plots based on the current slider values.
        """
        dim = self._get_dim()  # Get the dimensionality of the latent space
        n = self._N.get()  # Get the resolution for the mapping

        # Get the current slider values
        fixed_values = [var.get() for var in self.slider_vars]

        # Clear the existing plots
        for i in range(dim):
            for j in range(dim):
                self.ax_mapping_all[i, j].clear()

        # Generate the pairwise plots
        for j in range(dim):
            for i in range(dim):
                if i < j:
                    mesh, latent_points = self._plot_mapping(i,j)

                    # Set the fixed values for the other dimensions
                    for k in range(dim):
                        if k != i and k != j:
                            latent_points[:, k] = fixed_values[k]

                    # Predict the gain values
                    value_entry = self.gain_entry.get()
                    dataset = tf.data.Dataset.from_tensor_slices(latent_points).batch(256)
                    values = self.rna_gain[value_entry].predict(dataset, verbose=0).reshape((n, n))
                    values = self.gain_norm[value_entry] * (values)

                    # Plot the values
                    im = self.ax_mapping_all[j, i].pcolormesh(mesh[0], mesh[1], values, cmap='viridis')
                    self.ax_mapping_all[j, i].set_xlabel(f'Dim {i}')
                    self.ax_mapping_all[j, i].set_ylabel(f'Dim {j}')
                    # Update or create the colorbar
                    if hasattr(self, 'cbar'):
                        self.ax_mapping_all_cbar.update_normal(im)  # Update the existing colorbar
                    else:
                        self.ax_mapping_all_cbar = plt.colorbar(im, ax=self.ax_mapping_all[j, i])  # Create a new colorbar


                else:
                    # Hide the plots for i >= j
                    self.ax_mapping_all[j, i].axis("off")

        # Redraw the canvas
        self.canvas_mapping_all.draw()
    
    def on_click(self, event):
        
        if len(self._area) == 2:
            self._area.pop(0)
        self._area.append([event.xdata, event.ydata])
        
        if event.inaxes == self.ax_main or event.inaxes == self.ax_mapping:
            coord =  [event.xdata, event.ydata] 
            self.plot_detail(coord)
            
    def on_press(self, event):
        self.drawing = True
        self.update_area(event)

    def on_motion(self, event):
        if self.drawing:
            self.update_area(event)

    def on_release(self, event):
        self.drawing = False

    def update_area(self, event):
        if event.inaxes == self.ax_main or event.inaxes == self.ax_mapping:
            coord = [event.xdata, event.ydata]
            self.plot_detail(coord)