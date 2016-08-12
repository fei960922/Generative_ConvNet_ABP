function [audioCell, fs] = read_sound( config )
% read the audio .wav file

files = dir([config.inPath '*.wav']);
numAudio = 0;

if numAudio ~= length(files) || config.forceLearn == true;
    audioCell = cell(1, length(files));
    for iAudio = 1:length(files)
        [sound, fs] = audioread(fullfile(config.inPath, files(iAudio).name));
        audioCell{iAudio} = single(sound);
    end
end

end

