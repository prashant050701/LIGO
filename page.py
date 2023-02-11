import streamlit as st
import warnings warnings.filterwarnings('ignore',category=DeprecationWarning) st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests, os
from gwpy.timeseries import TimeSeries from gwosc.locate import get_urls
from gwosc import datasets
from gwosc.api import fetch_event_json
from copy import deepcopy import base64
import numpy as np
import tensorflow as tf from PIL import Image import matplotlib as mpl
mpl.use("agg")
from matplotlib.backends.backend_agg import RendererAgg _lock = RendererAgg.lock
def preprocessing(pixel):
pixel = pixel.resize((224, 224))
pixel = pixel.convert('RGB')
pixel = np.array(pixel)
pixel = pixel.reshape((1, 224, 224, 3)) pixel = pixel / 255.
return pixel
def getresult(pixel):
pixel = preprocessing(pixel)
#plt.imshow(pixel[0])
#st.pyplot()
tfmodel = tf.keras.models.load_model('/Users/user/Downloads/LIGO/vggmodel.h5')
#st.markdown(tfmodel.input)
res = tfmodel.predict(pixel)
res = list(res[0])
res = res.index(max(res))
res_map = {0:"Chirp",1:"Violin_Mode",2:"Koi_Fish",3:"Blip"}
return (res_map[res], res)
apptitle = 'Classification of LIGO Gravitational Wave data using Machine Learing' st.set_page_config(page_title=apptitle, page_icon=":eyeglasses:")
detectorlist = ['H1', 'L1', 'V1']
st.title('Classification of LIGO Gravitational Wave data using Machine Learning')
@st.cache(ttl=3600, max_entries=10) def load_gw(t0, detector, fs=4096):
strain = TimeSeries.fetch_open_data(detector, t0 - 14, t0 + 14, sample_rate=fs, cache=False) return strain
@st.cache(ttl=3600, max_entries=10) def get_eventlist():
allevents = datasets.find_datasets(type='events') eventset = set()
for ev in allevents:
name = fetch_event_json(ev)['events'][ev]['commonName'] if name[0:2] == 'GW':
eventset.add(name) eventlist = list(eventset) eventlist.sort()
return eventlist
st.sidebar.markdown("## Data Fetcher") eventlist = get_eventlist()
select_event = st.sidebar.selectbox('Finding Method',['By event name', 'By GPS'])
if select_event == 'By GPS':
str_t0 = st.sidebar.text_input('GPS Time', '1126259462.4') t0 = float(str_t0)
else:
chosen_event = st.sidebar.selectbox('Select Event', eventlist) t0 = datasets.event_gps(chosen_event)
detectorlist = list(datasets.event_detectors(chosen_event)) detectorlist.sort()
st.subheader(chosen_event)
st.write('GPS:', t0)
# Experiment to display masses
try:
jsoninfo = fetch_event_json(chosen_event)
for name, nameinfo in jsoninfo['events'].items():
st.write('Mass 1:', nameinfo['mass_1_source'], 'M$_{\odot}$') st.write('Mass 2:', nameinfo['mass_2_source'], 'M$_{\odot}$') st.write('Network SNR:', int(nameinfo['network_matched_filter_snr'])) st.write('\n')
except: pass
detector = st.sidebar.selectbox('Detector', detectorlist)
fs = 4096
maxband = 2000
high_fs = st.sidebar.checkbox('Full sample rate data') if high_fs:
fs = 16384 maxband = 8000
# -- Create sidebar for plot controls
st.sidebar.markdown('## Set Plot Parameters')
dtboth = st.sidebar.slider('Time Range (seconds)', 0.1, 8.0, 1.0) # min, max, default dt = dtboth / 2.0
freqrange = st.sidebar.slider('Band-pass frequency range (Hz)', min_value=10, max_value=maxband, value=(30, 400))
# -- Create sidebar for Q-transform controls
st.sidebar.markdown('#### Q-tranform plot')
vmax = st.sidebar.slider('Colorbar Max Energy', 10, 500, 25) # min, max, default qcenter = st.sidebar.slider('Q-value', 5, 120, 5) # min, max, default
qrange = (int(qcenter * 0.8), int(qcenter * 1.2))
# -- Create a text element and let the reader know the data is loading. strain_load_state = st.text('Loading data...this may take a minute') try:
strain_data = load_gw(t0, detector, fs) except:
st.warning(
'{0} data are not available for time {1}. Please try a different time and detector pair.'.format(detector, t0))
st.stop()
strain_load_state.text('Loading data...done!')
# -- Make a time series plot
cropstart = t0 - 0.2 cropend = t0 + 0.1
cropstart = t0 - dt cropend = t0 + dt
st.subheader('Raw data') center = int(t0)
strain = deepcopy(strain_data)
with _lock:
fig1 = strain.crop(cropstart, cropend).plot() # fig1 = cropped.plot()
st.pyplot(fig1, clear_figure=True)
st.subheader('Whitened and Band-passed Data') white_data = strain.whiten()
bp_data = white_data.bandpass(freqrange[0], freqrange[1])
bp_cropped = bp_data.crop(cropstart, cropend)
with _lock:
fig3 = bp_cropped.plot() st.pyplot(fig3, clear_figure=True)
st.subheader('Q-transform')
hq = strain.q_transform(outseg=(t0 - dt, t0 + dt), qrange=qrange)
with _lock:
fig4 = hq.plot()
ax = fig4.gca()
cbar = fig4.colorbar(label="Normalised energy", vmax=vmax, vmin=0) ax.grid(False)
ax.set_yscale('log')
ax.set_ylim(bottom=15)
st.pyplot(fig4, clear_figure=False)
ax.set_axis_off()
cbar.remove()
#szwst.pyplot(fig4, clear_figure=False) fig4.savefig('photo.png',bbox_inches='tight',pad_inches = 0)
img = Image.open('photo.png') #st.pyplot(img)
#print(fig4)
st.subheader("Classification")
st.markdown("""
""")
output = getresult(img)
st.markdown("The above spectrogram most likely belongs to "+ output[0] + " class.")
hide_streamlit_style = """ <style>
#MainMenu {visibility: hidden;} footer {visibility: hidden;} </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
