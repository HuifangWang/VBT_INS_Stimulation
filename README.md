# VBT_INS_Stimulation
VBT for Sitmulation in epilepsy

Codes: Virtual brain twins for stimulation in epilepsy
Huifang E Wang，Borana Dollomaja，Paul Triebkorn,  Gian Marco Duma, Adam Williamson, Julia Makhalova，Jean-Didier Lemarechal， Fabrice Bartolomei，Viktor Jirsa

Abstract:
Estimating the epileptogenic zone network (EZN) is an important part of the diagnosis of drug-resistant focal epilepsy and plays a pivotal role in treatment and intervention. Virtual brain twins provide a modeling method for personalized diagnosis and treatment. They integrate patient-specific brain topography with structural connectivity from anatomical neuroimaging such as MRI, and dynamic activity from functional recordings such as EEG and stereo-EEG (SEEG). Seizures demonstrate rich spatial and temporal features in functional recordings, which can be exploited to estimate the EZN. Stimulation-induced seizures can provide important and complementary information. We consider invasive SEEG stimulation and non-invasive temporal interference (TI) stimulation as a complementary approach. This paper offers a virtual brain twin framework for EZN diagnosis based on stimulation-induced seizures. It provides an important methodological and conceptual basis to make the transition from invasive to non-invasive diagnosis and treatment of drug-resistant focal epilepsy.


Dependencies: 

• FreeSurfer 6. 3.0: Used for volumetric segmentation and cortical surface reconstruction; https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
Cortical surface parcellation: https://github.com/HuifangWang/VEP_atlas_shared.git
• Mrtrix 0.3.16 software package for processing DW-MRI data. https://mrtrix.readthedocs.io/en/0.3.16/
• GARDEL 1.0 (Graphical user interface for Automatic Registration and Depth Electrodes Localization) for location of the SEEG contacts from post-implantation CT scans. https://meg.univ-amu.fr/doku.php?id=epitools:gardel
• SIMNIBS 4.0: Used for electric field calculation.; https://simnibs.github.io/simnibs/build/html/index.html
• Brainstorm3:Forward solution for scalp-EEG signals; https://neuroimage.usc.edu/brainstorm/Installation#Requirements
•	FSL v5.0.11 (fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL)
•	Stan v.2.21.0 (mc-stan.org)
•	make
•	A conda environment with following packages installed
o	Python 
o	Numpy
o	Scipy 
o	Nibabel
o	Pandas
o	MNE
o	Matplotlib
o	TVB library


1.	Structural scaffold reconstruction:
https://github.com/ins-amu/tvb-pipeline
This part of the code contains all functionality to reconstruct the structural scaffold of the personalized VEP model. Cortical surface and subcortical areas are reconstructed from a T1 weighted MRI image. Diffusion weighted MRI and tractography are used to calculate the structural connectome. Post-implantation brain imaging is used to extract the implanted SEEG contact locations and compute the projection of source activity to sensor signal. 
The code contains : 
•	process & dataflow in a Makefile
•	supporting Python utilities in a util module
Basic usage requires invoking make from within the directory that contains the file Makefile with a subject name and your dataset,
	make SUBJECT=tvb T1=data/t1 DWI=data/dwi fs-recon conn
where arguments are provided in ARG=value form, and outputs are given as names like fs-recon to perform the FreeSurfer recon-all -all reconstruction. See the following Targets section for a list of available outputs.
Targets
•	fs-recon: FreeSurfer reconstruction. Consists mainly of running recon-all -all. Uses T1.
•	conn: Connectivity matrices in text format. Uses T1 and DWI.
•	tvb: TVB zipfile, cortical and subcortical surfaces in TVB formats, region mappings. Uses T1 and DWI.
•	Elec: Positions of the contacts of depth electrodes and gain matrices. Uses T1, DWI, ELEC, and either ELEC_ENDPOINTS or ELEC_POS_GARDEL.
Instructions, on how the raw data should be ordered can be found in /template_reconstruction
Read the Makefile to see further modifiable options.

2.	VEP Stimulation workflow pipeline: 
#https://github.com/HuifangWang/VEP_INS_Stimulation
This part of the code contains functionality to simulate the VEP model, carry out Bayesian inference using sampling algorithms and perform virtual surgery. 
The directory examples/ contains the jupyter notebooks to run samping pipelines and plot the time-series and posterior results.
•	The directory vep/ contains the code for preprocessing the data, simulation, inference, analysis, and visualization.
•	The directiory 'data/' contains the data used for definition of the parcellation.


