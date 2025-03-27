# ULTra MoCap Processing

> Scripts, notebooks, and media assets

---

## 📹 Media Gallery

### 🔍 Diagrams

<div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 20px; margin-bottom: 30px;">
  <!-- <div style="flex: 1; min-width: 300px; text-align: center;">
    <img src="media/marker_and_sensor.png" alt="Marker and Sensor Layout" style="max-width: 100%;">
    <p><strong>Marker + Sensor Layout</strong><br>Placement of IMUs and markers</p>
  </div> -->
  <div style="flex: 1; min-width: 300px; text-align: center;">
    <img src="media/movement_diagram.png" alt="Movement Diagram" style="max-width: 100%;">
    <p><strong>Movement Diagram</strong><br>High-level schematic of task types</p>
  </div>
</div>

### 🎞️ Movement Videos

<div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 20px; margin-bottom: 30px;">
    <div style="text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 10px; border-radius: 5px; flex: 1; min-width: 200px;">
        <img src="media/armswing.gif" alt="Arm Swing" style="width: 100%; aspect-ratio: 9/16; object-fit: cover;">
        <h4>Arm Swing</h4>
    </div>
    <div style="text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 10px; border-radius: 5px; flex: 1; min-width: 200px;">
        <img src="media/crossbody.gif" alt="Cross Body Reach" style="width: 100%; aspect-ratio: 9/16; object-fit: cover;">
        <h4>Cross Body Reach</h4>
    </div>
    <div style="text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 10px; border-radius: 5px; flex: 1; min-width: 200px;">
        <img src="media/elbowflex.gif" alt="Elbow Flexion" style="width: 100%; aspect-ratio: 9/16; object-fit: cover;">
        <h4>Elbow Flexion</h4>
    </div>
    <div style="text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 10px; border-radius: 5px; flex: 1; min-width: 200px;">
        <img src="media/overheadreach.gif" alt="Overhead Reach" style="width: 100%; aspect-ratio: 9/16; object-fit: cover;">
        <h4>Overhead Reach</h4>
    </div>
    <div style="text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 10px; border-radius: 5px; flex: 1; min-width: 200px;">
        <img src="media/shoulderrotation.gif" alt="Shoulder Rotation" style="width: 100%; aspect-ratio: 9/16; object-fit: cover;">
        <h4>Shoulder Rotation</h4>
    </div>
</div>

---

## 📂 Processing Scripts

<div class="scripts-container" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(450px, 1fr)); gap: 20px;">

  <div class="script-card" style="border: 1px solid #e1e4e8; border-radius: 6px; padding: 16px; margin-bottom: 20px;">
    <h3><a href="processing/batch_IK.ipynb">batch_IK.ipynb</a> 📊</h3>
    <p>Performs batch processing for OpenSim inverse kinematics (IK).</p>
    <ul>
      <li>Data loading and transformation</li>
      <li>IK model execution</li>
      <li>Result visualization</li>
    </ul>
  </div>

  <div class="script-card" style="border: 1px solid #e1e4e8; border-radius: 6px; padding: 16px; margin-bottom: 20px;">
    <h3><a href="processing/batch_processH5.py">batch_processH5.py</a> 🔄</h3>
    <p>Combines all subject HDF5 files into a single file.</p>
    <ul>
      <li>Reads from multiple <code>.h5</code> datasets</li>
      <li>Cleans and validates data</li>
      <li>Merges into a unified structure</li>
    </ul>
  </div>

  <div class="script-card" style="border: 1px solid #e1e4e8; border-radius: 6px; padding: 16px; margin-bottom: 20px;">
    <h3><a href="processing/clean_sensors.py">clean_sensors.py</a> 🧹</h3>
    <p>Cleans raw sensor <code>.csv</code> files by removing unused channels.</p>
    <ul>
      <li>Removes Electromyography (sEMG)</li>
      <li>Removes Magnetometer data</li>
    </ul>
  </div>

  <div class="script-card" style="border: 1px solid #e1e4e8; border-radius: 6px; padding: 16px; margin-bottom: 20px;">
    <h3><a href="processing/movement_type_classifier.ipynb">movement_type_classifier.ipynb</a> 🤖</h3>
    <p>Trains a classifier to predict movement types from time-series sensor data.</p>
    <ul>
      <li>Feature extraction</li>
      <li>Model training and evaluation</li>
      <li>Accuracy and confusion matrix analysis</li>
    </ul>
  </div>

  <div class="script-card" style="border: 1px solid #e1e4e8; border-radius: 6px; padding: 16px; margin-bottom: 20px;">
    <h3><a href="processing/visualize_subject_distributions.py">visualize_subject_distributions.py</a> 👥</h3>
    <p>Analyzes subject demographics.</p>
    <ul>
      <li>Height</li>
      <li>Age</li>
      <li>Weight</li>
    </ul>
  </div>

  <div class="script-card" style="border: 1px solid #e1e4e8; border-radius: 6px; padding: 16px; margin-bottom: 20px;">
    <h3><a href="processing/visualize_speed_distributions.py">visualize_speed_distributions.py</a> 📈</h3>
    <p>Plots joint angular speeds to analyze movement profiles.</p>
    <ul>
      <li>Speed histograms</li>
      <li>Joint-specific movement trends</li>
    </ul>
  </div>

</div>