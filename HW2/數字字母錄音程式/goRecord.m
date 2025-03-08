% Recordings of digits and letters.
% Roger Jang, 20020328, 20050314, 20070320

% ====== Check MATLAB version
desiredVersion='8';
matlabVersion=version;
if ~strcmp(matlabVersion(1), desiredVersion)
	fprintf('Suggested MATLAB version: %s\n', desiredVersion);
end

% ====== Recording parameters
duration = 2;
fs = 16000;
%nbits = 8; format = 'uint8';		% 8-bit
nbits = 16; format = 'int16';		% 16-bit
waveDir = '../waveFile';

if exist(waveDir, 'dir') ~= 7,
	fprintf('Making wave directory => %s\n', waveDir);
	mkdir(waveDir);
end

validInput=0;
while ~validInput
	name = input('Please enter your student ID (for instance, 921510): ', 's');
	validName = name(find(name>=double('0')));		% 保留英文和數字
	if length(validName)>=2, validInput=1; end
end
userDir = [waveDir, '/', validName]; 
if exist(userDir, 'dir') ~= 7,
	fprintf('Making personal wave directory => %s\n', userDir);
	mkdir(userDir);
end

% ====== Get all digits/letters
allDigit = textread('digitLetter.txt', '%s', 'delimiter', '\n', 'whitespace', '');
% ====== Find recorded digits/letters
waveData=dir([userDir, '/*.wav']);
for i=1:length(waveData)
	waveData(i).mainName=waveData(i).name(1:2);
end
if ~isempty(waveData)
	fprintf('We have found %g recordings in %s.\n', length(waveData), userDir);
	recordedDigit={waveData.mainName};
	digitToRecord=setdiff(allDigit, recordedDigit);
else
	digitToRecord=allDigit;
end

digitNum=length(digitToRecord);
digitToRecord=digitToRecord(randperm(digitNum));	% Random permutation, since setdiff will also sort digitToRecord.
forcedRecording=0;
nextLoop=1;
for i=1:digitNum
	waveFile = [userDir, '/', digitToRecord{i}, '.wav'];
	displayText=digitToRecord{i}(1); 
	fprintf('%g/%g:\n', i, digitNum);
	while 1
		fileSaved=waveFileRecord(duration, fs, nbits, waveFile, displayText, forcedRecording);
		if fileSaved==0; break; end		% Skip recording
		ok=input('	Accept (Enter) or record again (n & Enter): ', 's');
		if isempty(ok)		% Recording completed
			fprintf('\tIf recording is not good, you can simply delete "%s".\n', waveFile);
			break;
		end
		delete(waveFile);
	end
end

fprintf('Congratulations! You have finished the recordings!\n');