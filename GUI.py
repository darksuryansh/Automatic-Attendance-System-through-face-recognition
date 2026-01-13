import cv2
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from deepface import DeepFace
from face_recognition import face_encodings, face_distance
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import time as time_module
import ast  # For safely evaluating string literals

# Configuration
ENCODINGS_FILE = 'data/encodings.pkl'
ATTENDANCE_FILE = 'data/attendance.csv'
TIMETABLE_FILE = 'data/timetable.csv'
SPOOF_THRESHOLD = 0.7
RECOGNITION_THRESHOLD = 0.5
DETECTOR_BACKEND = "opencv"
ATTENDANCE_INTERVAL = timedelta(hours=1)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
PROCESS_EVERY_N_FRAMES = 3

# Color Theme
BG_COLOR = "#2c3e50"
FG_COLOR = "#ecf0f1"
ACCENT_COLOR = "#3498db"
WARNING_COLOR = "#e74c3c"
SUCCESS_COLOR = "#2ecc71"
FRAME_COLOR = "#34495e"

class FaceRecognitionThread(threading.Thread):
    def __init__(self, video_capture, encodings, names):
        threading.Thread.__init__(self)
        self.video_capture = video_capture
        self.known_encodings = encodings
        self.known_names = names
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue()
        self.running = True
        self.frame_count = 0
        
    def run(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                self.frame_count += 1
                
                if self.frame_count % PROCESS_EVERY_N_FRAMES != 0:
                    continue
                
                try:
                    face_objs = DeepFace.extract_faces(
                        img_path=frame,
                        detector_backend=DETECTOR_BACKEND,
                        anti_spoofing=True,
                        enforce_detection=False
                    )
                except Exception as e:
                    print(f"Face detection error: {e}")
                    face_objs = []
                
                results = []
                for face_obj in face_objs:
                    x, y, w, h = face_obj["facial_area"]["x"], face_obj["facial_area"]["y"], \
                                 face_obj["facial_area"]["w"], face_obj["facial_area"]["h"]
                    
                    is_real = face_obj["is_real"] and (face_obj["antispoof_score"] >= SPOOF_THRESHOLD)
                    
                    if is_real:
                        face_roi = frame[y:y+h, x:x+w]
                        face_encoding = self.extract_face_encodings(face_roi)
                        
                        if face_encoding is not None:
                            name, confidence = self.recognize_face(face_encoding)
                            results.append({
                                "coords": (x, y, w, h),
                                "label": f"{name} ({confidence:.2f})" if name else "Unknown",
                                "color": (0, 255, 0) if name else (0, 255, 255),
                                "name": name,
                                "is_real": True
                            })
                    else:
                        results.append({
                            "coords": (x, y, w, h),
                            "label": f"Spoof ({face_obj['antispoof_score']:.2f})",
                            "color": (0, 0, 255),
                            "name": None,
                            "is_real": False
                        })
                
                self.result_queue.put((frame, results))
                
            except queue.Empty:
                continue
                
    def extract_face_encodings(self, face_img):
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        encodings = face_encodings(rgb_face)
        return encodings[0] if encodings else None
    
    def recognize_face(self, face_encoding):
        distances = face_distance(self.known_encodings, face_encoding)
        best_match_idx = np.argmin(distances)
        min_distance = distances[best_match_idx]
        
        if min_distance <= RECOGNITION_THRESHOLD:
            return self.known_names[best_match_idx], 1 - min_distance
        return None, None
    
    def stop(self):
        self.running = False

class EnhancedFaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Face Recognition System")
        self.root.geometry("1200x800")
        self.root.configure(bg=BG_COLOR)
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()
        
        self.known_encodings, self.known_names = self.load_encodings()
        self.attendance_df = self.load_attendance()
        self.timetable_df = self.load_timetable()
        self.recently_recognized = {}
        self.cap = None
        self.current_class = None
        self.running = False
        self.recognition_thread = None
        
        self.setup_gui()
        self.update_current_class()
        
    def configure_styles(self):
        self.style.configure('TFrame', background=BG_COLOR)
        self.style.configure('TLabel', background=BG_COLOR, foreground=FG_COLOR)
        self.style.configure('TButton', background=ACCENT_COLOR, foreground=FG_COLOR, 
                           font=('Helvetica', 10, 'bold'))
        self.style.configure('TNotebook', background=BG_COLOR)
        self.style.configure('TNotebook.Tab', background=FRAME_COLOR, foreground=FG_COLOR,
                           padding=[10, 5], font=('Helvetica', 10, 'bold'))
        self.style.map('TNotebook.Tab', background=[('selected', ACCENT_COLOR)])
        self.style.configure('Treeview', background=FRAME_COLOR, foreground=FG_COLOR,
                           fieldbackground=FRAME_COLOR)
        self.style.configure('Treeview.Heading', background=ACCENT_COLOR, foreground=FG_COLOR)
        self.style.map('Treeview', background=[('selected', '#2980b9')])
        
    def setup_gui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.live_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.live_tab, text="Live Recognition")
        self.setup_live_tab()
        
        self.log_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.log_tab, text="Attendance Log")
        self.setup_log_tab()
        
        self.stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_tab, text="Statistics")
        self.setup_stats_tab()
        
        self.status_var = tk.StringVar()
        self.status_var.set("System Ready")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, 
                                 anchor=tk.W, bg=FRAME_COLOR, fg=FG_COLOR, font=('Helvetica', 10))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_live_tab(self):
        main_frame = ttk.Frame(self.live_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        video_frame = ttk.LabelFrame(left_frame, text="Camera Feed", padding=10)
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = ttk.Button(control_frame, text="▶ Start", command=self.start_camera,
                                  style='Accent.TButton')
        self.start_btn.pack(side=tk.LEFT, padx=5, ipadx=10, ipady=5)
        
        self.stop_btn = ttk.Button(control_frame, text="■ Stop", command=self.stop_camera,
                                 state=tk.DISABLED, style='Accent.TButton')
        self.stop_btn.pack(side=tk.LEFT, padx=5, ipadx=10, ipady=5)
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10)
        
        class_card = ttk.LabelFrame(right_frame, text="Current Class", padding=10)
        class_card.pack(fill=tk.X, pady=5)
        
        self.class_var = tk.StringVar()
        self.class_var.set("No class currently scheduled")
        class_label = ttk.Label(class_card, textvariable=self.class_var, 
                              font=('Helvetica', 12, 'bold'), foreground=ACCENT_COLOR)
        class_label.pack(pady=5)
        
        info_frame = ttk.Frame(class_card)
        info_frame.pack(fill=tk.X)
        
        ttk.Label(info_frame, text="Time:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.time_var = tk.StringVar()
        self.time_var.set("--:-- -- --:-- --")
        ttk.Label(info_frame, textvariable=self.time_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(info_frame, text="Expected:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.expected_var = tk.StringVar()
        self.expected_var.set("--")
        ttk.Label(info_frame, textvariable=self.expected_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(info_frame, text="Present:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.present_var = tk.StringVar()
        self.present_var.set("--")
        present_label = ttk.Label(info_frame, textvariable=self.present_var, foreground=SUCCESS_COLOR)
        present_label.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(info_frame, text="Absent:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.absent_var = tk.StringVar()
        self.absent_var.set("--")
        absent_label = ttk.Label(info_frame, textvariable=self.absent_var, foreground=WARNING_COLOR)
        absent_label.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        
        list_frame = ttk.LabelFrame(right_frame, text="Student Status", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        canvas = tk.Canvas(list_frame, bg=FRAME_COLOR, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.student_frame = scrollable_frame
        
    def setup_log_tab(self):
        main_frame = ttk.Frame(self.log_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        filter_frame = ttk.Frame(main_frame)
        filter_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(filter_frame, text="Filter by:").pack(side=tk.LEFT, padx=5)
        
        self.filter_name_var = tk.StringVar()
        name_entry = ttk.Entry(filter_frame, textvariable=self.filter_name_var, width=20)
        name_entry.pack(side=tk.LEFT, padx=5)
        name_entry.bind("<Return>", lambda e: self.apply_filters())
        
        ttk.Label(filter_frame, text="Date:").pack(side=tk.LEFT, padx=5)
        
        self.filter_date_var = tk.StringVar()
        date_entry = ttk.Entry(filter_frame, textvariable=self.filter_date_var, width=10)
        date_entry.pack(side=tk.LEFT, padx=5)
        date_entry.bind("<Return>", lambda e: self.apply_filters())
        
        filter_btn = ttk.Button(filter_frame, text="Apply Filter", command=self.apply_filters)
        filter_btn.pack(side=tk.LEFT, padx=5)
        
        reset_btn = ttk.Button(filter_frame, text="Reset", command=self.reset_filters)
        reset_btn.pack(side=tk.LEFT, padx=5)
        
        log_frame = ttk.LabelFrame(main_frame, text="Attendance Records", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        tree_frame = ttk.Frame(log_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.tree = ttk.Treeview(tree_frame, columns=('Name', 'Timestamp', 'Status'), 
                               show='headings', selectmode='browse')
        
        self.tree.heading('Name', text='Name')
        self.tree.heading('Timestamp', text='Timestamp')
        self.tree.heading('Status', text='Status')
        
        self.tree.column('Name', width=150)
        self.tree.column('Timestamp', width=180)
        self.tree.column('Status', width=100)
        
        y_scroll = ttk.Scrollbar(tree_frame, orient='vertical', command=self.tree.yview)
        x_scroll = ttk.Scrollbar(tree_frame, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        
        self.tree.grid(row=0, column=0, sticky='nsew')
        y_scroll.grid(row=0, column=1, sticky='ns')
        x_scroll.grid(row=1, column=0, sticky='ew')
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        export_frame = ttk.Frame(main_frame)
        export_frame.pack(fill=tk.X, pady=5)
        
        export_csv_btn = ttk.Button(export_frame, text="Export to CSV", command=self.export_to_csv)
        export_csv_btn.pack(side=tk.LEFT, padx=5)
        
        export_pdf_btn = ttk.Button(export_frame, text="Export to PDF", command=self.export_to_pdf)
        export_pdf_btn.pack(side=tk.LEFT, padx=5)
        
        self.update_attendance_log()
        
    def setup_stats_tab(self):
        main_frame = ttk.Frame(self.stats_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        stats_frame = ttk.LabelFrame(main_frame, text="Attendance Statistics", padding=10)
        stats_frame.pack(fill=tk.BOTH, expand=True)
        
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(8, 4), facecolor=FRAME_COLOR)
        self.ax.set_facecolor(FRAME_COLOR)
        self.fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=stats_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        controls_frame = ttk.Frame(stats_frame)
        controls_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(controls_frame, text="Time Period:").pack(side=tk.LEFT, padx=5)
        
        self.stats_period_var = tk.StringVar(value="week")
        periods = [("Last Week", "week"), ("Last Month", "month"), ("Last Year", "year"), ("All Time", "all")]
        
        for text, value in periods:
            rb = ttk.Radiobutton(controls_frame, text=text, variable=self.stats_period_var, 
                                value=value, command=self.update_stats)
            rb.pack(side=tk.LEFT, padx=5)
        
        summary_frame = ttk.LabelFrame(main_frame, text="Summary Statistics", padding=10)
        summary_frame.pack(fill=tk.BOTH, pady=10)
        
        stats_grid = ttk.Frame(summary_frame)
        stats_grid.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(stats_grid, text="Total Students:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.total_students_var = tk.StringVar(value="0")
        ttk.Label(stats_grid, textvariable=self.total_students_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(stats_grid, text="Avg. Attendance:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.avg_attendance_var = tk.StringVar(value="0%")
        ttk.Label(stats_grid, textvariable=self.avg_attendance_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(stats_grid, text="Most Punctual:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.most_punctual_var = tk.StringVar(value="None")
        ttk.Label(stats_grid, textvariable=self.most_punctual_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(stats_grid, text="Least Punctual:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.least_punctual_var = tk.StringVar(value="None")
        ttk.Label(stats_grid, textvariable=self.least_punctual_var).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Initial update
        self.update_stats()
        
    def update_stats(self):
        """Update the statistics display based on selected time period"""
        period = self.stats_period_var.get()
        now = datetime.now()
        
        # Filter data based on time period
        if period == "week":
            start_date = now - timedelta(days=7)
            df = self.attendance_df[self.attendance_df['timestamp'] >= start_date]
        elif period == "month":
            start_date = now - timedelta(days=30)
            df = self.attendance_df[self.attendance_df['timestamp'] >= start_date]
        elif period == "year":
            start_date = now - timedelta(days=365)
            df = self.attendance_df[self.attendance_df['timestamp'] >= start_date]
        else:
            df = self.attendance_df.copy()
        
        if df.empty:
            self.ax.clear()
            self.ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            self.canvas.draw()
            
            self.total_students_var.set("0")
            self.avg_attendance_var.set("0%")
            self.most_punctual_var.set("None")
            self.least_punctual_var.set("None")
            return
        
        # Plot attendance by day
        self.ax.clear()
        
        # Group by date and count unique students
        daily_attendance = df.groupby(df['timestamp'].dt.date)['name'].nunique()
        daily_attendance.plot(kind='bar', ax=self.ax, color=ACCENT_COLOR)
        
        self.ax.set_title(f"Daily Attendance ({period.capitalize()})", color=FG_COLOR)
        self.ax.set_xlabel("Date", color=FG_COLOR)
        self.ax.set_ylabel("Number of Students", color=FG_COLOR)
        self.ax.tick_params(colors=FG_COLOR)
        
        # Rotate x-axis labels for better readability
        for label in self.ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')
        
        self.canvas.draw()
        
        # Update summary statistics
        unique_students = df['name'].nunique()
        total_classes = df['class'].nunique()
        
        if total_classes > 0:
            avg_attendance = (len(df) / (unique_students * total_classes)) * 100
        else:
            avg_attendance = 0
        
        # Calculate punctuality (students with most and least attendance)
        student_attendance = df['name'].value_counts()
        
        if not student_attendance.empty:
            most_punctual = student_attendance.idxmax()
            least_punctual = student_attendance.idxmin()
        else:
            most_punctual = "None"
            least_punctual = "None"
        
        self.total_students_var.set(str(unique_students))
        self.avg_attendance_var.set(f"{avg_attendance:.1f}%")
        self.most_punctual_var.set(most_punctual)
        self.least_punctual_var.set(least_punctual)
    
    def load_encodings(self):
        """Load face encodings from file"""
        if os.path.exists(ENCODINGS_FILE):
            with open(ENCODINGS_FILE, 'rb') as f:
                data = pickle.load(f)
                return data['encodings'], data['names']
        return [], []
    
    def load_attendance(self):
        """Load attendance records from CSV"""
        if os.path.exists(ATTENDANCE_FILE):
            df = pd.read_csv(ATTENDANCE_FILE)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Add class column if not present (for backward compatibility)
            if 'class' not in df.columns:
                df['class'] = 'No Class'
            return df
        return pd.DataFrame(columns=['name', 'timestamp', 'status', 'class'])
    
    def load_timetable(self):
        """Load timetable data"""
        if os.path.exists(TIMETABLE_FILE):
            df = pd.read_csv(TIMETABLE_FILE)
            df['start_time'] = pd.to_datetime(df['start_time']).dt.time
            df['end_time'] = pd.to_datetime(df['end_time']).dt.time
            return df
        return pd.DataFrame(columns=['class_name', 'day', 'start_time', 'end_time', 'students'])
    
    def update_current_class(self):
        """Update the current class based on timetable and current time"""
        now = datetime.now()
        current_time = now.time()
        current_day = now.strftime("%A")
        
        # Find classes happening now
        current_classes = self.timetable_df[
            (self.timetable_df['day'] == current_day) &
            (self.timetable_df['start_time'] <= current_time) &
            (self.timetable_df['end_time'] >= current_time)
        ]
        
        if not current_classes.empty:
            self.current_class = current_classes.iloc[0]
            self.class_var.set(self.current_class['class_name'])
            
            # Format time display
            start_str = self.current_class['start_time'].strftime("%I:%M %p")
            end_str = self.current_class['end_time'].strftime("%I:%M %p")
            self.time_var.set(f"{start_str} to {end_str}")
            
            # Update expected students
            try:
                expected_students = ast.literal_eval(self.current_class['students'])
                self.expected_var.set(f"{len(expected_students)} students")
            except:
                expected_students = []
                self.expected_var.set("Error reading student list")
            
            # Update present/absent counts
            self.update_attendance_counts()
        else:
            self.current_class = None
            self.class_var.set("No class currently scheduled")
            self.time_var.set("--:-- -- --:-- --")
            self.expected_var.set("--")
            self.present_var.set("--")
            self.absent_var.set("--")
        
        # Update student list
        self.update_student_list()
        
        # Schedule next update
        self.root.after(60000, self.update_current_class)
    
    def update_attendance_counts(self):
        """Update the present/absent counts for the current class"""
        if self.current_class is None:
            return
            
        now = datetime.now()
        today = now.date()
        class_name = self.current_class['class_name']
        
        try:
            expected_students = ast.literal_eval(self.current_class['students'])
        except:
            expected_students = []
        
        # Get attendance for this class today
        class_attendance = self.attendance_df[
            (self.attendance_df['class'] == class_name) &
            (self.attendance_df['timestamp'].dt.date == today)
        ]
        
        present_students = class_attendance['name'].unique()
        present_count = len(present_students)
        absent_count = len(expected_students) - present_count
        
        self.present_var.set(f"{present_count} students")
        self.absent_var.set(f"{absent_count} students")
    
    def update_student_list(self):
        """Update the list of students in the current class"""
        for widget in self.student_frame.winfo_children():
            widget.destroy()
            
        if self.current_class is None:
            ttk.Label(self.student_frame, text="No class currently scheduled").pack(pady=10)
            return
            
        try:
            expected_students = ast.literal_eval(self.current_class['students'])
        except:
            expected_students = []
            ttk.Label(self.student_frame, text="Error reading student list").pack(pady=10)
            return
            
        now = datetime.now()
        today = now.date()
        class_name = self.current_class['class_name']
        
        # Get attendance for this class today
        class_attendance = self.attendance_df[
            (self.attendance_df['class'] == class_name) &
            (self.attendance_df['timestamp'].dt.date == today)
        ]
        
        present_students = class_attendance['name'].unique()
        
        # Create student cards
        for i, student in enumerate(expected_students):
            frame = ttk.Frame(self.student_frame, relief=tk.RIDGE, padding=5)
            frame.pack(fill=tk.X, pady=2)
            
            # Status indicator
            status = "Present" if student in present_students else "Absent"
            color = SUCCESS_COLOR if status == "Present" else WARNING_COLOR
            
            ttk.Label(frame, text=student, width=20).pack(side=tk.LEFT, padx=5)
            ttk.Label(frame, text=status, foreground=color, width=10).pack(side=tk.LEFT, padx=5)
            
            # Last seen time if present
            if status == "Present":
                last_seen = class_attendance[class_attendance['name'] == student]['timestamp'].max()
                last_seen_str = last_seen.strftime("%I:%M %p")
                ttk.Label(frame, text=last_seen_str).pack(side=tk.RIGHT, padx=5)
    
    def update_attendance_log(self):
        """Update the attendance log Treeview"""
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        name_filter = self.filter_name_var.get().lower()
        date_filter = self.filter_date_var.get()
        
        filtered_df = self.attendance_df.copy()
        
        if name_filter:
            filtered_df = filtered_df[filtered_df['name'].str.lower().str.contains(name_filter)]
            
        if date_filter:
            try:
                filter_date = pd.to_datetime(date_filter).date()
                filtered_df = filtered_df[filtered_df['timestamp'].dt.date == filter_date]
            except:
                pass
        
        # Add records to treeview
        for _, row in filtered_df.iterrows():
            timestamp_str = row['timestamp'].strftime("%Y-%m-%d %I:%M %p")
            self.tree.insert("", tk.END, values=(
                row['name'],
                timestamp_str,
                row['status']
            ))
    
    def apply_filters(self):
        """Apply filters to attendance log"""
        self.update_attendance_log()
    
    def reset_filters(self):
        """Reset all filters"""
        self.filter_name_var.set("")
        self.filter_date_var.set("")
        self.update_attendance_log()
    
    def export_to_csv(self):
        """Export attendance data to CSV"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save attendance data as CSV"
        )
        
        if filename:
            try:
                self.attendance_df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Attendance data exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data: {str(e)}")
    
    def export_to_pdf(self):
        """Export attendance data to PDF"""
        messagebox.showinfo("Info", "PDF export functionality would be implemented here")
    
    def start_camera(self):
        """Start the camera and face recognition thread"""
        if self.cap is not None:
            return
            
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            self.cap = None
            return
            
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Start recognition thread
        self.recognition_thread = FaceRecognitionThread(self.cap, self.known_encodings, self.known_names)
        self.recognition_thread.start()
        
        # Start video update loop
        self.update_video()
        
    def stop_camera(self):
        """Stop the camera and recognition thread"""
        if self.cap is None:
            return
            
        self.running = False
        if self.recognition_thread is not None:
            self.recognition_thread.stop()
            self.recognition_thread.join()
            self.recognition_thread = None
            
        self.cap.release()
        self.cap = None
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        # Clear video display
        self.video_label.config(image='')
    
    def update_video(self):
        """Update the video feed with processed frames"""
        if not self.running or self.cap is None:
            return
            
        ret, frame = self.cap.read()
        
        if ret:
            display_frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
            
            if self.recognition_thread is not None:
                try:
                    self.recognition_thread.frame_queue.put_nowait(display_frame)
                except queue.Full:
                    pass
                
                try:
                    processed_frame, results = self.recognition_thread.result_queue.get_nowait()
                    
                    for result in results:
                        x, y, w, h = result['coords']
                        cv2.rectangle(processed_frame, (x, y), (x+w, y+h), result['color'], 2)
                        cv2.putText(processed_frame, result['label'], (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, result['color'], 2)
                        
                        if result['name'] and result['is_real']:
                            self.record_attendance(result['name'])
                    
                    display_frame = processed_frame
                except queue.Empty:
                    pass
            
            img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        self.root.after(30, self.update_video)
    
    def record_attendance(self, name):
        """Record attendance for recognized student"""
        now = datetime.now()
        today = now.date()
        
        if name in self.recently_recognized:
            last_time = self.recently_recognized[name]
            if now - last_time < ATTENDANCE_INTERVAL:
                return
        
        if self.current_class is not None:
            class_name = self.current_class['class_name']
            try:
                expected_students = ast.literal_eval(self.current_class['students'])
            except:
                expected_students = []
            
            if name in expected_students:
                status = "Present"
            else:
                status = "Unexpected"
        else:
            class_name = "No Class"
            status = "Unexpected"
        
        new_record = {
            'name': name,
            'timestamp': now,
            'status': status,
            'class': class_name
        }
        
        self.attendance_df = pd.concat([
            self.attendance_df,
            pd.DataFrame([new_record])
        ], ignore_index=True)
        
        self.save_attendance()
        
        self.update_attendance_counts()
        self.update_student_list()
        self.update_attendance_log()
        self.update_stats()
        
        self.recently_recognized[name] = now
        self.status_var.set(f"Recorded attendance for {name} in {class_name}")
    
    def save_attendance(self):
        """Save attendance data to CSV"""
        try:
            os.makedirs(os.path.dirname(ATTENDANCE_FILE), exist_ok=True)
            self.attendance_df.to_csv(ATTENDANCE_FILE, index=False)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save attendance: {str(e)}")
    
    def on_closing(self):
        """Handle window closing event"""
        self.stop_camera()
        self.save_attendance()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedFaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()