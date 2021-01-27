//===- Mem2Reg.cpp - The -mem2reg pass, a wrapper around the Utils lib ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is a simple pass wrapper around the PromoteMemToReg function call
// exposed by the Utils library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include <vector>

using namespace llvm;

// Custom imports
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SHA1.h"
#include "llvm/Support/Compression.h"
#include <unordered_map>


#define DEBUG_TYPE "mem2reg"

STATISTIC(NumPromoted, "Number of alloca's promoted");

static bool promoteMemoryToRegister(Function &F, DominatorTree &DT,
                                    AssumptionCache &AC) {
  std::vector<AllocaInst *> Allocas;
  BasicBlock &BB = F.getEntryBlock(); // Get the entry node for the function
  bool Changed = false;

  while (true) {
    Allocas.clear();

    // Find allocas that are safe to promote, by looking at all instructions in
    // the entry node
    for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I)
      if (AllocaInst *AI = dyn_cast<AllocaInst>(I)) // Is it an alloca?
        if (isAllocaPromotable(AI))
          Allocas.push_back(AI);

    if (Allocas.empty())
      break;

    PromoteMemToReg(Allocas, DT, &AC);
    NumPromoted += Allocas.size();
    Changed = true;
  }
  return Changed;
}

PreservedAnalyses PromotePass::run(Function &F, FunctionAnalysisManager &AM) {
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  if (!promoteMemoryToRegister(F, DT, AC))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

namespace {

struct PromoteLegacyPass : public FunctionPass {
  // Pass identification, replacement for typeid
  static char ID;

  PromoteLegacyPass() : FunctionPass(ID) {
    initializePromoteLegacyPassPass(*PassRegistry::getPassRegistry());
  }
  
  std::string hexStr(const char *data, int len) {     
    constexpr char hexmap[] = {'0', '1', '2', '3', '4', '5', '6', '7',     
                             '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};

    std::string s(len * 2, ' ');     
    for (int i = 0; i < len; ++i) {     
      s[2 * i]     = hexmap[(data[i] & 0xF0) >> 4];     
      s[2 * i + 1] = hexmap[data[i] & 0x0F];     
    }     
    return s;     
  }

  std::string hashString(StringRef S) {     
    SHA1 Hasher;     
    Hasher.update(S);     
    StringRef Hexed = Hasher.final();     
    return hexStr(Hexed.data(), Hexed.size());     
  }

  bool usedBy(Function* F, Value* V, std::unordered_map<Value*, bool> &memo) {
    if(memo.find(V) != memo.end()) {
      return memo[V];
    }

    for(User *U : V->users()) {
      if(auto *I = dyn_cast<Instruction>(U)) {
        if(I->getFunction()->getName() == F->getName()) {
          memo[V] = true;
          return true;
        }
      } 
      if(usedBy(F, U, memo)) {
        return true;
      }
    }

    memo[V] = false;
    return false;
  }

  // runOnFunction - To run this pass, first we calculate the alloca
  // instructions that are safe for promotion, then we promote each one.
  bool runOnFunction(Function &F_to_serialize) override {
    auto mod = CloneModule(*F_to_serialize.getParent());

    // get all functions that are not F and store names
    std::vector<Function*> functions;
    for(auto &func : *mod) {
      if(func.getName() != F_to_serialize.getName()) {
        functions.push_back(&func);
      }
    }

    // go through all functions in the module except F
    for(auto func : functions) {

      // if F uses func, delete body, else erase it
      bool used_by_F_to_serialize = false;
      for(User *U : func->users()) {
        if(Instruction* call = dyn_cast<Instruction>(U)) {
          Function* caller = call->getParent()->getParent();
          if(caller->getName() == F_to_serialize.getName()) {
            func->deleteBody();
            used_by_F_to_serialize = true;
            break;
          }
        }
      }

      // F_to_serialize does not call func
      if(!used_by_F_to_serialize) {
        func->replaceAllUsesWith(UndefValue::get(func->getType())); 
        func->eraseFromParent();
      }
    }

    std::vector<GlobalVariable*> unused_globals;
    for(auto &G : mod->globals()) { // iterate over global variables in module
      bool used_by_F_to_serialize = false;

      std::unordered_map<Value*, bool> memo;
      for(User *U : G.users()) {
        if(auto *I = dyn_cast<Instruction>(U)) {
          if(I->getFunction()->getName() == F_to_serialize.getName()) {
            G.setInitializer(NULL); 
            used_by_F_to_serialize = true;
            break; 
          }
        } 
        if(usedBy(&F_to_serialize, U, memo)) {
          used_by_F_to_serialize = true;
          break;
        }
      }

      if(!used_by_F_to_serialize) {
        unused_globals.push_back(&G);
      }
    }

    // remove all global variables not used by F_to_serialize
    for(auto global_var : unused_globals) {
      global_var->replaceAllUsesWith(UndefValue::get(global_var->getType())); 
      global_var->eraseFromParent();
    }

    std::string Data;
    raw_string_ostream OS(Data);
    WriteBitcodeToFile(*mod, OS);
    SmallVector<char> buff;
    auto error = zlib::compress(StringRef(Data), buff, 1);
    if(error) {
      abort();
    }

    Twine toHash = mod->getName() + F_to_serialize.getName();
    std::string hashed = hashString(StringRef(toHash.str()));
    std::string file ="/Users/peyton/UROP/CloudCompiler/data/compressed/Mem2Reg/" + hashed + ".csv";
    StringRef fileName(file);

    std::error_code EC;
    raw_fd_ostream fdOS(fileName, EC, llvm::sys::fs::OF_None);

    fdOS << mod->getName() << "," << F_to_serialize.getName() << "," << "Mem2Reg," << buff.size() << "\n";

    if (skipFunction(F_to_serialize))
      return false;

    DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    AssumptionCache &AC =
        getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F_to_serialize);
    return promoteMemoryToRegister(F_to_serialize, DT, AC);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.setPreservesCFG();
  }
};

} // end anonymous namespace

char PromoteLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(PromoteLegacyPass, "mem2reg", "Promote Memory to "
                                                    "Register",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(PromoteLegacyPass, "mem2reg", "Promote Memory to Register",
                    false, false)

// createPromoteMemoryToRegister - Provide an entry point to create this pass.
FunctionPass *llvm::createPromoteMemoryToRegisterPass() {
  return new PromoteLegacyPass();
}
