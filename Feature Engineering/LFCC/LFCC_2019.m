clear; close all; clc;

% set paths to the wave files and protocols
pathToFeatures = "LFCCFeatures";

% read train protocol
trainfileID = fopen("LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt");
trainprotocol = textscan(trainfileID, '%s%s%s%s%s');
fclose(trainfileID);
trainfilelist = trainprotocol{2};  
% read eval protocol
devfileID = fopen("LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt");
devprotocol = textscan(devfileID, '%s%s%s%s%s');
fclose(devfileID);
devfilelist = devprotocol{2}; 
% read eval protocol
evalfileID = fopen("LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.eval.trl.txt");
evalprotocol = textscan(evalfileID, '%s%s%s%s%s');
fclose(evalfileID);
evalfilelist = evalprotocol{2}; 


%% Feature extraction 
% extract features for training data and store them
disp('Extracting features for data...');

for i=1:length(evalfilelist)
    filePath = fullfile("LA\ASVspoof2019_LA_eval\flac",horzcat(evalfilelist{i},'.flac'));
    [x,fs] = audioread(filePath);
    [stat,delta,double_delta] = extract_lfcc(x,fs,20,512,20);
    LFCC = [stat delta double_delta]';
    filename_LFCC = fullfile("2019LA_LFCC\LFCCFeatures_2019LA_mat\eval",horzcat('LFCC_', evalfilelist{i}, '.mat'))
    parsave(filename_LFCC, LFCC)
    LFCC = [];
end
disp('Done!');


%% supplementary function
function parsave(fname, x)
    save(fname, 'x')
end
