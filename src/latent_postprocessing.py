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
import threading
from tkinter import messagebox
from abc import ABC, abstractmethod

class PostprocessingBase(ABC):
    def __init__(self, root, data_loader):
        self.root = root
        self.data_loader = data_loader
        self._initialize_config()
        self._initialize_model_components()
        self._setup_ui()


    @abstractmethod
    def get_label(self):
        """Get the name of the label and value
        
        returns:
        --------
        label: str
        value: numpy array
        """
        pass 
    
    @abstractmethod    
    def plot_detail(self, coord):
        """Routine to plot decoded.

        Args:
            coord (_type_): _description_
        """
        pass
    
    # @abstractmethod    
    # def get_latent_space(self):
    #     """Get the latent space
        
    #     returns:
    #     --------
    #     numpy array
    #     """
    #     pass    
    
    @abstractmethod
    def add_custom_buttons(self, parent):
        """Add custom buttons to the control frame."""
        pass
    
    @abstractmethod
    def add_settings(self,parent):
        """Add custom settings to the control frame."""
        pass    
    
    def _initialize_config(self):
        """Load and initialize configuration."""
        self.config = get_config(self.data_loader.result_folder + '/conf.json')
        self.filtered = self.config.get("filter", {})  # Use .get() to avoid KeyError
    
    def _initialize_model_components(self):
        """Initialize model-related components."""
        self.latent_space = self.data_loader.model["latent_space"]
        self.encoder = self.data_loader.model["encoder"]
        self.decoder = self.data_loader.model["decoder"]
        self.filtered = self.config["filter"]
        self.vae_norm = self.data_loader.vae_norm
        self.dim = self.latent_space.shape[1]
        self._area = []

    def _setup_ui(self):
        """Set up the Tkinter UI."""
        self.root.title("Interactive Visualization")
        self._setup_control_frame()
        self._setup_plot_frame()
        self._setup_detail_window()
        self._bind_events()
        self.plot_main()

    def _setup_control_frame(self):
        """Set up the control frame with widgets."""
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        self._add_axis_selection(control_frame)
        self._add_pca_settings(control_frame)
        self._add_grid_size_setting(control_frame)
        self._add_buttons(control_frame)
        self.add_settings(control_frame)

    def _add_spinbox(self, parent, variable, command):
        """Helper function to add a spinbox."""
        spinbox = ttk.Spinbox(parent, from_=0, to=self.dim - 1, textvariable=variable, command=getattr(self, command), width=3)
        spinbox.pack(side=tk.TOP)
        
    def _add_axis_selection(self, parent):
        """Add X and Y axis selection widgets."""
        tk.Label(parent, text="X-axis:").pack(side=tk.TOP)
        self.x_axis_var = tk.IntVar(value=0)
        self._add_spinbox(parent, self.x_axis_var, "plot_main")

        tk.Label(parent, text="Y-axis:").pack(side=tk.TOP)
        self.y_axis_var = tk.IntVar(value=1)
        self._add_spinbox(parent, self.y_axis_var, "plot_main")

    def _add_pca_settings(self, parent):
        """Add PCA-related settings."""
        tk.Label(parent, text="PCA dim:").pack(side=tk.TOP)
        self._pca_dim = tk.IntVar(value=2)
        tk.Entry(parent, textvariable=self._pca_dim, width=3).pack(side=tk.TOP)

        tk.Label(parent, text="Enable PCA:").pack(side=tk.TOP)
        self.enablePCA = tk.BooleanVar(value=False)
        tk.Checkbutton(parent, text="Enable", variable=self.enablePCA, command=self.plot_main).pack(side=tk.TOP)

    def _add_grid_size_setting(self, parent):
        """Add grid size setting."""
        tk.Label(parent, text="N:").pack(side=tk.TOP)
        self._N = tk.IntVar(value=50)
        tk.Entry(parent, textvariable=self._N, width=3).pack(side=tk.TOP)

    def _add_buttons(self, parent):
        """Add action buttons."""
        self.add_custom_buttons(parent)       
        tk.Button(self.root, text="Quit", command=self.quit_app).pack(pady=10)

    def _setup_plot_frame(self):
        """Set up the main plot frame."""
        fig_frame = tk.Frame(self.root)
        fig_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig_main, self.ax_main = plt.subplots()
        self.canvas_main = FigureCanvasTkAgg(self.fig_main, master=fig_frame)
        self.canvas_main.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_detail_window(self):
        """Set up the detail plot window."""
        self.detail_window = tk.Toplevel(self.root)
        self.detail_window.title("Detail Plot")
        self.fig_detail, self.ax_detail = plt.subplots()
        self.canvas_detail = FigureCanvasTkAgg(self.fig_detail, master=self.detail_window)
        self.canvas_detail.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _bind_events(self):
        """Bind mouse events to the canvas."""
        self.canvas_main.mpl_connect("button_press_event", self.on_click)
        self.canvas_main.mpl_connect("button_press_event", self.on_press)
        self.canvas_main.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas_main.mpl_connect("button_release_event", self.on_release)
        self.drawing = False  # Track if the mouse is pressed
        
    def quit_app(self):
        print("Application is closing...")  # For debugging/cleanup purposes
        self.root.quit()
        self.root.destroy() 

        
    def plot_main(self):
        self.ax_main.clear()
        
        _, gg = self.get_label()
        
        self._pca = PCA(n_components=self._pca_dim.get())

        if self.enablePCA.get():
            latent_space = self._pca.fit_transform(self.latent_space)
            explained_var = self._pca.explained_variance_ratio_
            sc = self.ax_main.scatter(latent_space[:, self.x_axis_var.get()], latent_space[:, self.y_axis_var.get()], c = gg)
            self.ax_main.set_xlabel(f"Dimension {self.x_axis_var.get()} ({explained_var[self.x_axis_var.get()]:.2f})")
            self.ax_main.set_ylabel(f"Dimension {self.y_axis_var.get()} ({explained_var[self.y_axis_var.get()]:.2f})")

        else:
            sc = self.ax_main.scatter(self.latent_space[:, self.x_axis_var.get()], self.latent_space[:, self.y_axis_var.get()], c = gg)
            self.ax_main.set_xlabel(f"Dimension {self.x_axis_var.get()}")
            self.ax_main.set_ylabel(f"Dimension {self.y_axis_var.get()}")
            
        if hasattr(self, 'scatter_cb'):
            self.scatter_cb.update_normal(sc)
        else:
            self.scatter_cb = self.fig_main.colorbar(sc, ax=self.ax_main)
        self.canvas_main.draw()

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
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class PostprocessingFCI(PostprocessingBase):
    def __init__(self, root, data_loader: DataLoader):
        super().__init__(root, data_loader)

    # def get_latent_space(self):
    def _initialize_model_components(self):
        
        super()._initialize_model_components()
        self.rna_gain = self.data_loader.model["latent_gain"]
        self.gain =  self.data_loader.dataset['values']
        self.time =  self.data_loader.dataset['time']
        
        key = list(self.filtered.keys())[0]
        gain_val = np.array(self.gain[key])
        print(len(gain_val))
        mask = gain_val >= self.filtered[key]

        for key in self.gain.keys():
            self.gain[key] = np.array(self.gain[key])[mask]
            
        self.x_max = None
        self.gain_norm = self.data_loader.gain_norm
        self.vae_norm = self.data_loader.vae_norm     
        
    def add_custom_buttons(self, parent):
        tk.Button(parent, text="Mapping", command=self.plot_mapping).pack(pady=10)
        tk.Button(parent, text="Optimize", command=self.find_best).pack(pady=10)
        tk.Button(parent, text="Slices", command=self.show_latent_space).pack(pady=5)

    def get_label(self):
        return self.gain_entry.get(), self.gain[self.gain_entry.get()]

    def add_settings(self, parent):
        options = list(self.gain.keys())
        tk.Label(parent, text="Select value").pack(side=tk.TOP)
        self.gain_entry = tk.StringVar(value="gain") # Default to heatmap based on gain
        self.gain_entry.trace_add("write", self.plot_main)
        gain_entry_menu = tk.OptionMenu(parent, self.gain_entry, *options)
        gain_entry_menu.pack(side=tk.TOP)

    def plot_detail(self, coord):
        # Create a detailed plot in the second figure based on clicked coordinates
        self.ax_detail.clear()
        
        gain_entry = self.gain_entry.get()
        
        if self.enablePCA.get():
            dim = self._pca_dim.get()
        else:
            dim = self.dim
        
        axis = []
        
        for i in list(set(range(dim)) - {self.x_axis_var.get(), self.y_axis_var.get()}):
            axis.append(i)
        
        latent_point = np.zeros((1,dim))
        latent_point[0,self.x_axis_var.get()] = coord[0]
        latent_point[0,self.y_axis_var.get()] = coord[1]
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
        mesh, unfit = self._plot_mapping(self.x_axis_var.get(),self.y_axis_var.get())            
        
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