
name="070418a";
channel="ch09";
pathin="C:/Users/bioAMD/Desktop/Nathan/Tracking/saved_predictions/pred_modele_divisions4";
pixelxy="1.367";
pixelz="1.37";
slices="104";


File.makeDirectory(pathin+"/"+name+"/"); 
 //setBatchMode(true);
 
all_files = false;
tstart = 301;
tstop = 360;

files = getFileList(pathin);	

if (all_files) {
	for (i=0; i < files.length; i++) {
		
		if (endsWith(files[i], channel+".tif") == false ) {
			continue;
		}
		
		filename = files[i];
		
		s=0; ind = "";
		while (substring(filename, name.length+2+s, name.length + 3 +s) != "_") {
			s += 1;
		}
		j = substring(filename, name.length + 2, name.length + 2 + s);
		ind_j = 0;
		while (substring(j, ind_j, ind_j+1) == "0") {
			ind_j += 1;
			ind += "0";  
		}
		
		
		processFile( substring(j, ind.length), ind );
	}
}
else { 
	for(t=tstart; t < tstop; t++) {
		entier = d2s(t,0);
		if(t>= 100) {i=""; processFile(entier, i); }
		else if (t>=10) { i="0"; processFile(entier, i);}
		else if (t>=0) { i="00"; processFile(entier, i);}
		else {print("erreur: t < 0";}
	}
} 




function processFile(i,digit) {
open(pathin+"/"+name+"_t"+digit+i+"_"+channel+".tif");
//run("Brightness/Contrast...");
//run("Despeckle", "stack");
//run("Enhance Contrast...", "saturated=0.35 normalize process_all use");
//setMinAndMax("0", "255");

//run("Enhance Contrast", "saturated=0.35");
//setMinAndMax(0, 255);
//run("Enhance Contrast", "saturated=0.35 normalize process_all use");
//run("Apply LUT", "stack");
//run("Enhance Contrast", "saturated=0.35");
//run("8-bit");

//run("Brightness/Contrast...");
run("Enhance Contrast...", "saturated=0 normalize process_all");
run("8-bit");
run("Apply LUT", "stack");
run("Properties...", "channels=1 slices="+slices+" frames=1 pixel_width="+pixelxy+" pixel_height="+pixelxy+" voxel_depth="+pixelz+"");
run("Save BioEmergences VTK...", "save="+pathin+"/"+name+"/"+name+"_t"+digit+i+"_"+channel+".vtk.gz");
close();

	}

