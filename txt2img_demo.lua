
require 'image'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'lfs'
torch.setdefaulttensortype('torch.FloatTensor')

local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
local dict = {}
for i = 1,#alphabet do
    dict[alphabet:sub(i,i)] = i
end
ivocab = {}
for k,v in pairs(dict) do
  ivocab[v] = k
end

opt = {
  filenames = 'scripts/val_filename.txt',
  dataset = 'coco',
  batchSize = 8,        -- number of samples to produce
  noisetype = 'normal',  -- type of noise distribution (uniform / normal).
  imsize = 1,            -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px, ...
  noisemode = 'random',  -- random / line / linefull1d / linefull
  gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
  display = 0,           -- Display image: 0 = false, 1 = true
  nz = 100,              
  doc_length = 201,
  queries = 'scripts/val_captions.txt',
  checkpoint_dir = 'ckpts',
  net_gen = 'coco_fast_t70_nc3_nt128_nz100_bs64_cls0.5_ngf196_ndf196_100_net_G.t7',
  net_txt = 'coco_gru18_bs64_cls0.5_ngf128_ndf128_a10_c512_80_net_T.t7',
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

noise = torch.Tensor(opt.batchSize, opt.nz, opt.imsize, opt.imsize)
net_gen = torch.load(opt.checkpoint_dir .. '/' .. opt.net_gen)
net_txt = torch.load(opt.checkpoint_dir .. '/' .. opt.net_txt)
if net_txt.protos ~=nil then net_txt = net_txt.protos.enc_doc end

net_gen:evaluate()
net_txt:evaluate()

-- -- Extract all text features.
-- local fea_txt = {}
-- -- Decode text for sanity check.
-- local raw_txt = {}
-- local raw_img = {}


count = 1
all_queries = {}
for query_str in io.lines(opt.queries) do
  all_queries[count] = query_str
  count = count + 1
end
all_filenames = {}
count = 1
for filename_str in io.lines(opt.filenames) do
  all_filenames[count] = filename_str
  count = count + 1
end
-- print(#all_queries, all_queries[1])
-- print(#all_filenames, all_filenames[1])

num_queries = #all_queries

if opt.gpu > 0 then
  require 'cunn'
  require 'cudnn'
  net_gen:cuda()
  net_txt:cuda()
  noise = noise:cuda()
end

for i = 1,3 do 
  query_str = all_queries[i]
  filename_str = all_filenames[i]

  local txt = torch.zeros(1,opt.doc_length,#alphabet)
  for t = 1,opt.doc_length do
    local ch = query_str:sub(t,t)
    local ix = dict[ch]
    if ix ~= 0 and ix ~= nil then
      txt[{1,t,ix}] = 1
    end
  end
  txt = txt:cuda()
  fea_txt = net_txt:forward(txt):clone()
  print(txt:size(), fea_txt:size())
end

-- for query_str in all_queries do
--   local txt = torch.zeros(1,opt.doc_length,#alphabet)
--   for t = 1,opt.doc_length do
--     local ch = query_str:sub(t,t)
--     local ix = dict[ch]
--     if ix ~= 0 and ix ~= nil then
--       txt[{1,t,ix}] = 1
--     end
--   end
--   raw_txt[#raw_txt+1] = query_str
--   txt = txt:cuda()
--   fea_txt[#fea_txt+1] = net_txt:forward(txt):clone()
-- end

-- local html = '<html><body><h1>Generated Images</h1><table border="1px solid gray" style="width=100%"><tr><td><b>Caption</b></td><td><b>Image</b></td></tr>'

-- for i = 1,#fea_txt do
--   print(string.format('generating %d of %d', i, #fea_txt))
--   local cur_fea_txt = torch.repeatTensor(fea_txt[i], opt.batchSize, 1)
--   local cur_raw_txt = raw_txt[i]
--   if opt.noisetype == 'uniform' then
--     noise:uniform(-1, 1)
--   elseif opt.noisetype == 'normal' then
--     noise:normal(0, 1)
--   end
--   local images = net_gen:forward{noise, cur_fea_txt:cuda()}
--   local visdir = string.format('results/%s', opt.dataset)
--   lfs.mkdir('results')
--   lfs.mkdir(visdir)
--   local fname = string.format('%s/img_%d', visdir, i)
--   local fname_png = fname .. '.png'
--   local fname_txt = fname .. '.txt'
--   images:add(1):mul(0.5)
--   --image.save(fname_png, image.toDisplayTensor(images,4,torch.floor(opt.batchSize/4)))
--   image.save(fname_png, image.toDisplayTensor(images,4,opt.batchSize/2))
--   html = html .. string.format('\n<tr><td>%s</td><td><img src="%s"></td></tr>',
--                                cur_raw_txt, fname_png)
--   os.execute(string.format('echo "%s" > %s', cur_raw_txt, fname_txt))
-- end


