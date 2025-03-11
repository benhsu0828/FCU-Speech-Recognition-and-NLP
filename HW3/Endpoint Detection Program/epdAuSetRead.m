function auSet=epdAuSetRead(auDir)
% epdAuSetRead: Read all audio files from the given audio directory
%
%	Usage:
%		auSet=epdAuSetRead(auDir)
%
%	Description:
%		auSet=epdAuSetRead(auDir) returns a structure array containing data from all the audio files within a given audio directory
%			auDir: a given audio directory
%			auSet: the returned structure array
%
%	Example:
%		auDir='waveFile';
%		auSet=epdAuSetRead(auDir);
%		disp(auSet);

%	Roger Jang, 20060304, 20130331

if nargin<1, selfdemo; return; end

temp=recursiveFileList(auDir, 'wav');
auNum=length(temp);
if auNum==0, auSet=[]; return; end

for i=1:auNum
	auSet(i)=myAudioRead(temp(i).path);
end

% ====== Attach more fields
errorIndex=[];
for i=1:auNum
%	fprintf('%d/%d ===> %s\n', i, auNum, temp(i).path);
	auSet(i).speaker=temp(i).parentDir;		% Get speaker information
	[parentDir, mainName]=fileparts(temp(i).path);
	index=find(mainName=='_');
	if length(index)~=2, errorIndex=[errorIndex, i]; continue; end
	try
		auSet(i).epByHuman=[eval(mainName(index(1)+1:index(2)-1)), eval(mainName(index(2)+1:end))];
	catch
		errorIndex=[errorIndex, i];
	end
end

if ~isempty(errorIndex)
	fprintf('Removing %d files due to error in file name format...\n', length(errorIndex));
	auSet(errorIndex)=[];
	fprintf('List of files with wrong file name format:\n');
	for i=1:length(errorIndex)
		fprintf('%d/%d: file name = %s\n', i, length(errorIndex), temp(i).path);
	end
end

% ====== Self demo
function selfdemo
mObj=mFileParse(which(mfilename));
strEval(mObj.example);
